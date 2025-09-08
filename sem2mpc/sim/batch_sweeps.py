import os, json, time, csv
import casadi as ca
import numpy as np
from compiler.build_ocp import build_ocp

def run_once(task_json):
    nlp, meta = build_ocp(task_json)
    N, nx, nu = meta['N'], meta['nx'], meta['nu']
    lbg, ubg = meta['bounds']['lbg'], meta['bounds']['ubg']
    solver = ca.nlpsol('solver','ipopt',nlp, {'ipopt.print_level':0, 'print_time':0})
    x_init = ca.DM.zeros((nx, N+1)); u_init = ca.DM.zeros((nu,N))
    init_guess = ca.vertcat(ca.reshape(x_init,-1,1), ca.reshape(u_init,-1,1))
    t0=time.time()
    sol = solver(x0=init_guess, lbg=lbg, ubg=ubg)
    t1=time.time()
    x_opt = ca.reshape(sol['x'][:nx*(N+1)], nx, N+1).T.full()
    return x_opt, (t1-t0)

def sweep(task_path, out_csv='sweep_results.csv'):
    base = json.load(open(task_path,'r',encoding='utf-8'))
    rows=[['terminal_scale','control_w','horizon','obs_cx','delta_max','solve_time','end_err','min_dist']]
    for ts in [1,3,5]:
        for cw in [0.02,0.05,0.1]:
            for N in [30,50,70]:
                for cx in [1.0,2.0,2.5,3.0]:
                    for dmax in [0.5,0.3,0.2]:
                        conf = json.loads(json.dumps(base))
                        conf['terminal_scale']=ts
                        conf['weights']['control']=[cw,cw]
                        conf['horizon']=N
                        conf['obstacle']['center'][0]=cx
                        conf['constraints']['delta_max']=dmax
                        conf['constraints']['delta_min']=-dmax
                        tmp='_tmp.json'; open(tmp,'w',encoding='utf-8').write(json.dumps(conf))
                        try:
                            xs, t = run_once(tmp)
                            end_err = float(np.linalg.norm(xs[-1,:2]-np.array(conf['goal'][:2])))
                            d = np.sqrt((xs[:,0]-conf['obstacle']['center'][0])**2 + (xs[:,1]-conf['obstacle']['center'][1])**2)
                            min_d = float(np.min(d))
                        except Exception as e:
                            t = float('nan'); end_err=float('inf'); min_d=float('nan')
                        rows.append([ts,cw,N,cx,dmax,t,end_err,min_d])
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    print('✅ wrote', out_csv)

if __name__=='__main__':
    sweep('dsl/example_task_curve_01.json')
