import torch
import gzip
import pickle
import sys
import numpy as np
import os
import argparse
import time
from os import getenv
from os.path import join
from multiprocessing import Pool, Manager

PROJECTS_DIR = getenv('PROJECTS_DIR')
sys.path.insert(1, join(PROJECTS_DIR, 'BlackBoxOptimization'))
from utils import split_array, split_array_idx, get_split_indices
import logging
logging.basicConfig(level=logging.WARNING)

def uniform_sample(shape, bounds, device='cpu'):
    return (bounds[0] - bounds[1]) * torch.rand(shape, device=device) + bounds[1]
    
class ShipMuonShield():
    
    new_parametrization = {'HA': [0, 7, 8, 9, 10, 11, 12, 13, 14],
                       'M1': [1, 15, 16, 17, 18, 19, 20, 21, 22],
                       'M2': [2, 23, 24, 25, 26, 27, 28, 29, 30], #SC
                       'M3': [3, 31, 32, 33, 34, 35, 36, 37, 38],
                       'M4': [4, 39, 40, 41, 42, 43, 44, 45, 46],
                       'M5': [5, 47, 48, 49, 50, 51, 52, 53, 54],
                       'M6': [6, 55, 56, 57, 58, 59, 60, 61, 62]}

    sc_v6 = torch.as_tensor([231.00,  0., 353.08, 125.08, 184.83, 150.19, 186.81, 
         50.00,  50.00, 119.00, 119.00,   2.00,   2.00, 1.00, 0.00,
        0.,  0.,  0.,  0.,  0.,   0., 1., 0.,
        45.69,  45.69,  22.18,  22.18,  27.01,  16.24, 3.00, 0.00,
        0.,  0.,  0.,  0.,  0.,  0., 1., 0., 
        24.80,  48.76,   8.00, 104.73,  15.80,  16.78, 1.00, 0.00,
        3.00, 100.00, 192.00, 192.00,   2.00,   4.80, 1.00, 0.00,
        3.00, 100.00,   8.00, 172.73,  46.83,   2.00, 1.00, 0.00])

    hybrid_idx =  [new_parametrization['HA'][0]] + new_parametrization['M2'][:-2] + [new_parametrization['M3'][0]]+\
           new_parametrization['M4'] + new_parametrization['M5'] + new_parametrization['M6']
    
    DEFAULT_PHI = sc_v6

    def __init__(self,
                 W0:float = 1558731.375,
                 cores:int = 45,
                 n_samples:int = 0,
                 input_dist:float = 0.85,
                 sensitive_plane:float = 57,#distance between end of shield and sensplane
                 average_x:bool = True,
                 loss_with_weight:bool = True,
                 fSC_mag:bool = True,
                 simulate_fields:bool = False,
                 cavern:bool = False,
                 seed:int = None,
                 left_margin = 2,
                 right_margin = 2,
                 y_margin = 3,
                 dimensions_phi = 35,
                 ) -> None:
        
        self.left_margin = left_margin
        self.MUON = 13
        self.right_margin = right_margin
        self.y_margin = y_margin
        self.W0 = W0
        self.cores = cores
        self.muons_file = join(PROJECTS_DIR,'MuonsAndMatter/data/inputs.pkl')
        self.n_samples = n_samples
        self.input_dist = input_dist
        self.average_x = average_x
        self.loss_with_weight = loss_with_weight
        self.sensitive_plane = sensitive_plane
        self.sensitive_film_params = {'dz': 0.01, 'dx': 4, 'dy': 6,'position': sensitive_plane} #the center is in end of muon shield + position
        self.fSC_mag = fSC_mag
        self.simulate_fields = simulate_fields
        self.seed = seed
        self.dimensions_phi = dimensions_phi   
        self.cavern = cavern

        if dimensions_phi == 29: self.params_idx = self.fixed_sc
        elif dimensions_phi == 35: self.params_idx = self.hybrid_idx
        elif dimensions_phi == 72: self.params_idx = slice(None)
        self.DEFAULT_PHI = self.DEFAULT_PHI[self.params_idx]

        
        sys.path.insert(1, join(PROJECTS_DIR,'MuonsAndMatter/python/bin'))
        from run_simulation import run
        self.run_muonshield = run
        

    def sample_x(self,phi=None):
        with gzip.open(self.muons_file, 'rb') as f:
            x = pickle.load(f)
        if 0<self.n_samples<=x.shape[0]: x = x[:self.n_samples]
        return x

    def simulate(self,phi:torch.tensor,muons = None, return_nan = False): 
        phi = phi.flatten() 
        phi = self.add_fixed_params(phi)
        if muons is None: muons = self.sample_x()

        workloads = split_array(muons,self.cores)

        with Pool(self.cores) as pool:
            result = pool.starmap(self.run_muonshield, 
                                  [(workload,phi.cpu().numpy(),self.input_dist,True,self.fSC_mag,self.sensitive_film_params,self.cavern,
                                    self.simulate_fields,None,return_nan,self.seed, False) for workload in workloads])
        all_results = []
        for rr in result:
            resulting_data,weight = rr
            if resulting_data.size == 0: continue
            all_results += [resulting_data]
        if len(all_results) == 0:
            all_results = [[np.nan]*8]
        all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device,dtype=torch.get_default_dtype())
        all_results[3],all_results[4],all_results[5] = self.propagate_to_sensitive_plane(*all_results[:6])
        return *all_results, torch.as_tensor(weight,device = phi.device)

    def muon_loss(self,x,y,particle, weight = None):
        charge = -1*torch.sign(particle)
        mask = (-charge*x <= self.left_margin) & (-self.right_margin <= -charge*x) & (torch.abs(y) <= self.y_margin) & ((torch.abs(particle).to(torch.int))==self.MUON)
        x = x[mask]
        charge = charge[mask]
        loss = torch.sqrt(1 + (charge*x-self.right_margin)/(self.left_margin+self.right_margin)) 
        if weight is not None:
            weight = weight[mask]
            loss *= weight
        return loss
    def get_total_length(self,phi):
        length = 0
        for m,idx in self.parametrization.items():
            if m == '?': continue
            params = phi[idx]
            length += params[0]
        return length
    def get_weight(self,phi,density = 7.874E-3):
        '''Get the weight of the muon shield.
        phi: torch.tensor with the parameters of the muon shield.
        density: density of iron in kg/cm^3'''
        def volume_block(dz,dx1,dx2,dy1,dy2):
            return dz*(dx1*dy1+dx2*dy2)/2
        phi = self.add_fixed_params(phi)
        volume = 0
        for m,idx in self.parametrization.items():
            params = phi[idx]
            Ymgap = 5 if self.fSC_mag and m == 'M2' else 0
            if m == '?': continue
            if self.fSC_mag and m in ['M1', 'M3']: continue
            volume += volume_block(2*params[0],params[1],params[2],params[3],params[4]) #core
            volume += volume_block(2*params[0],params[1]*params[7],params[2]*params[7],params[3]+Ymgap,params[4]+Ymgap) #lateral yoke
            volume += volume_block(2*params[0],params[1]+params[5]+params[1]*params[7]+params[8],params[2]+params[6]+params[2]*params[7]+params[8],params[1]*params[7], params[2]*params[7]) #upper yoke
        return 4*volume*density #4 due to symmetry
    def weight_loss(self,W,beta = 10):
        return 1+torch.exp(beta*(W-self.W0)/self.W0)
    def calc_loss(self,px,py,pz,x,y,z,particle,W = None):
        loss = self.muon_loss(x,y,particle).sum()+1
        if self.loss_with_weight:
            loss *= self.weight_loss(W)
            loss = torch.where(W>3E6,1e8,loss)
        return loss.to(torch.get_default_dtype())
    def __call__(self,phi,muons = None):
        if phi.dim()>1:
            y = []
            for p in phi:
                y.append(self(p))
            return torch.stack(y)
        px,py,pz,x,y,z,particle,W = self.simulate(phi,muons)
        return self.calc_loss(None,None,None,x,y,None,particle,W)
    
    def propagate_to_sensitive_plane(self,px,py,pz,x,y,z, epsilon = 1e-12):
        '''Deprecated'''
        return x,y,z
        z += self.sensitive_plane
        x += self.sensitive_plane*px/(pz+epsilon)
        y += self.sensitive_plane*py/(pz+epsilon)
        return x,y,z

    def GetBounds(self,zGap:float = 1.,device = torch.device('cpu'), correct_bounds = True):
        magnet_lengths = [(50 + zGap, 400 + zGap)] * 8  #previously 170-300
        dX_bounds = [(1, 100)] * 2
        dY_bounds = [(1, 200)] * 2 
        gap_bounds = [(2, 70)] * 2 
        yoke_bounds = [(0.25, 4)]
        inner_gap_bounds = [(0., 20.)]
        bounds = magnet_lengths + 8*(dX_bounds + dY_bounds + gap_bounds + yoke_bounds + inner_gap_bounds)
        if self.fSC_mag: 
            bounds[self.parametrization['M2'][4]] = (15,70)
            bounds[self.parametrization['M2'][5]] = (15,70)
        bounds = torch.tensor(bounds,device=device,dtype=torch.get_default_dtype()).T
        return bounds[:,self.params_idx]
    
    def add_fixed_params(self,phi:torch.Tensor):
        if phi.size(-1) != 72:
            new_phi = self.sc_v6.clone().to(phi.device)
            new_phi[torch.as_tensor(self.params_idx,device = phi.device)] = phi
            if self.fSC_mag:
                new_phi[self.parametrization['M2'][2]] = new_phi[self.parametrization['M2'][1]]
                new_phi[self.parametrization['M2'][4]] = new_phi[self.parametrization['M2'][3]]
        else: new_phi = phi
        return new_phi
    
    
    def sim(self,rank):
        
        muons = self.sample_x()
        tasks = split_array(muons, 3*32)
        task = tasks[(rank-1)*32:rank*32, :, :]
        
        return_nan = False
        field_map_file = None
        draw_magnet = False
        back_track = False
        phi =  self.sc_v6

        with Pool(32) as pool:
            # missing cavern
            result = pool.starmap(self.run_muonshield, [(task_cpu, phi.cpu().numpy(), self.input_dist, True, self.fSC_mag,
                                                         self.sensitive_film_params, self.simulate_fields, field_map_file, return_nan, self.seed, draw_magnet, back_track) for task_cpu in task])
        all_results = []

        for rr in result:
            resulting_data,weight = rr
            if len(resulting_data)==0: continue
            all_results += [resulting_data]
        print(f"Weight = {weight} kg")
        if len(all_results) != 0:
            all_results = np.concatenate(all_results, axis=0)
            #all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device,dtype=torch.get_default_dtype())
            
            x_tensor = torch.tensor([data['x'][-1] for data in all_results], dtype=torch.float32)
            y_tensor = torch.tensor([data['y'][-1] for data in all_results], dtype=torch.float32)
            pdg_id_tensor = torch.tensor([data['pdg_id'][-1] for data in all_results], dtype=torch.int32)

            # Optional: If you have weight data, convert it to tensor
            weight_tensor = torch.tensor([1.0] * len(all_results), dtype=torch.float32)  # Example weights

            
            # Compute the muon loss
            loss_muons = muon_shield.muon_loss(x_tensor, y_tensor, pdg_id_tensor, weight_tensor).sum() + 1
            loss = loss_muons * muon_shield.weight_loss(weight_tensor)
            loss = torch.where( weight_tensor > 3E6, 1e8, loss)
            
            result_data = {
                    'name': 'sc_v6',
                    'rank': rank,
                    'seed': all_results[0].get('seed', []),  
                    'weight': all_results[0].get('weight_total', 0),
                    'loss_muons': loss_muons,
                    'loss': loss,
                    'particle_type': [part.get('pdg_id', [])[-1] for part in all_results], 
                    'w': [part.get('w', None) for part in all_results],
                    'x': [part.get('x', []) for part in all_results],
                    'y': [part.get('y', []) for part in all_results],
                    'z': [part.get('z', []) for part in all_results],
                    'px': [part.get('px', []) for part in all_results],
                    'py': [part.get('py', []) for part in all_results],
                    'pz': [part.get('pz', []) for part in all_results],
                }
        else:
            result_data = {}
        return result_data
    
    
    def sim_new(self,rank):
        
        cores = 1
        cpu = 3
        n_task = cores*cpu
        
        muons = self.sample_x()
        tasks = split_array(muons, n_task)        
        task = tasks[(rank-1)*cores:rank*cores]
        
        return_nan = False
        field_map_file = None
        draw_magnet = False
        back_track = False
        phi =  self.sc_v6

        return_weight = False
        add_cavern = True



        
        
        with Pool(cores) as pool:
            result = pool.starmap(self.run_muonshield, [(task_cpu, phi.cpu().numpy(), self.input_dist, return_weight, self.fSC_mag,
                                                         self.sensitive_film_params, add_cavern, self.simulate_fields, field_map_file, return_nan, self.seed, draw_magnet) for task_cpu in task])

        all_results = []
        '''
        for rr in result:
            resulting_data,weight = rr
            if len(resulting_data)==0: continue
            all_results += [resulting_data]
        print(f"Weight = {weight} kg")
        '''
        if len(all_results) != 0:
            all_results = np.concatenate(all_results, axis=0)
            #all_results = torch.as_tensor(np.concatenate(all_results, axis=0).T,device = phi.device,dtype=torch.get_default_dtype())
            
            x_tensor = torch.tensor([data['x'][-1] for data in all_results], dtype=torch.float32)
            y_tensor = torch.tensor([data['y'][-1] for data in all_results], dtype=torch.float32)
            pdg_id_tensor = torch.tensor([data['pdg_id'][-1] for data in all_results], dtype=torch.int32)

            # Optional: If you have weight data, convert it to tensor
            weight_tensor = torch.tensor([1.0] * len(all_results), dtype=torch.float32)  # Example weights

            
            # Compute the muon loss
            loss_muons = muon_shield.muon_loss(x_tensor, y_tensor, pdg_id_tensor, weight_tensor).sum() + 1
            loss = loss_muons * muon_shield.weight_loss(weight_tensor)
            loss = torch.where( weight_tensor > 3E6, 1e8, loss)
            
            result_data = {
                    'name': 'sc_v6',
                    'rank': rank,
                    'seed': all_results[0].get('seed', []),  
                    'weight': all_results[0].get('weight_total', 0),
                    'loss_muons': loss_muons,
                    'loss': loss,
                    'particle_type': [part.get('pdg_id', [])[-1] for part in all_results], 
                    'w': [part.get('w', None) for part in all_results],
                    'x': [part.get('x', []) for part in all_results],
                    'y': [part.get('y', []) for part in all_results],
                    'z': [part.get('z', []) for part in all_results],
                    'px': [part.get('px', []) for part in all_results],
                    'py': [part.get('py', []) for part in all_results],
                    'pz': [part.get('pz', []) for part in all_results],
                }
        else:
            result_data = {}
        return result_data

def split_array_idx(phi, 
                    indices = None, 
                    num_splits = None,
                    N_samples = None,
                    file = None):
    if indices is None: indices = get_split_indices(num_splits,N_samples)
    phi = phi.view(-1,phi.size(-1))
    splits = []
    for p in phi:
        for idx in indices:
            input = [p,idx]
            if file is not None: input.append(file)
            splits.append(input)
    return splits

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--n_tasks_per_node", type=int, default=32)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--warm", dest='SC', action='store_false')
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--cluster", action='store_true')
    parser.add_argument("--field_map", action='store_true')
    args = parser.parse_args()

    # Determine rank using SLURM environment variables
    rank = int(os.environ.get('SLURM_PROCID'))
    size = int(os.environ.get('SLURM_NTASKS'))

    print(f"Process {rank} of {size} starting...")

    # Parameters
    t0 = time.time()
    seed = rank * 5  # Ensure different seeds per rank
    name = 'sc_v6'

    
    print(f"Process {rank} working on {name}")
    t1 = time.time()

    muon_shield = ShipMuonShield(
        cores=size,
        fSC_mag=args.SC,
        dimensions_phi=35,
        sensitive_plane=67,
        input_dist = None,
        simulate_fields=False,
        seed=seed
    )
    

    result_data = muon_shield.sim_new(rank)
    
    if result_data:
        try:
            output_directory = '/disk/users/gfrise/Project/BlackBoxOptimization/Outputs/'
            output_file = f"results_rank_{rank}.pkl"
            with open(output_directory + output_file, 'wb') as f:
                pickle.dump(result_data, f)
            print(f"Process {rank} finished and saved combined results to {output_file}")
        except Exception as e:
            print(f"Failed to save combined results: {e}")

    t2 = time.time()
    print(f"Process {rank} completed {name} in {t2 - t1} seconds")
    