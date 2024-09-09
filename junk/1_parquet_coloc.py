# --------------------- application related librairies and parameters ------------------

import os, shutil
import logging
from time import sleep, time

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime

from dask import compute

# from dask.delayed import delayed
from dask.distributed import performance_report, wait
from distributed.diagnostics import MemorySampler

# to force flushing of memory
import gc, ctypes

from glob import glob

from cstes import swot_dir, drifters_dir, get_proj, lonlat2xy, zarr_dir
from swot import browse_swot_250, add_mask_inside_swot, build_swath_polygon, partition_swot_interp_grad

import pynsitu as pyn

# ---- Run parameters

#root_dir = "/home/datawork-lops-osi/equinox/mit4320/parcels/"
run_name = "coloc_drifters_swot_250"

# will overwrite existing results
# overwrite = True
#overwrite = False

# dask parameters

dask_jobs = 1  # number of dask pbd jobs
jobqueuekw = dict(processes=25, cores=25)  # uplet debug

# ---------------------------- dask utils - do not touch -------------------------------


def spin_up_cluster(
    ctype,
    jobs=None,
    processes=None,
    fraction=0.8,
    timeout=20,
    **kwargs,
):
    """Spin up a dask cluster ... or not
    Waits for workers to be up for distributed ones

    Paramaters
    ----------
    ctype: None, str
        Type of cluster: None=no cluster, "local", "distributed"
    jobs: int, optional
        Number of PBS jobs
    processes: int, optional
        Number of processes per job (overrides default in .config/dask/jobqueue.yml)
    timeout: int
        Timeout in minutes is cluster does not spins up.
        Default is 20 minutes
    fraction: float, optional
        Waits for fraction of workers to be up

    """

    if ctype is None:
        return
    elif ctype == "local":
        from dask.distributed import Client, LocalCluster

        dkwargs = dict(n_workers=14, threads_per_worker=1)
        dkwargs.update(**kwargs)
        cluster = LocalCluster(**dkwargs)  # these may not be hardcoded
        client = Client(cluster)
    elif ctype == "distributed":
        from dask_jobqueue import PBSCluster
        from dask.distributed import Client

        assert jobs, "you need to specify a number of dask-queue jobs"
        cluster = PBSCluster(processes=processes, **kwargs)
        cluster.scale(jobs=jobs)
        client = Client(cluster)

        if not processes:
            processes = cluster.worker_spec[0]["options"]["processes"]

        flag = True
        start = time()
        while flag:
            wk = client.scheduler_info()["workers"]
            print("Number of workers up = {}".format(len(wk)))
            sleep(5)
            if len(wk) >= processes * jobs * fraction:
                flag = False
                print("Cluster is up, proceeding with computations")
            now = time()
            if (now - start) / 60 > timeout:
                flag = False
                print("Timeout: cluster did not spin up, closing")
                cluster.close()
                client.close()
                cluster, client = None, None

    return cluster, client


def dashboard_ssh_forward(client):
    """returns the command to execute on a local computer in order to
    have access to dashboard at the following address in a browser:
    http://localhost:8787/status
    """
    env = os.environ
    port = client.scheduler_info()["services"]["dashboard"]
    return f'ssh -N -L {port}:{env["HOSTNAME"]}:8787 {env["USER"]}@datarmor1-10g', port


def close_dask(cluster, client):
    logging.info("starts closing dask cluster ...")
    try:

        client.close()
        logging.info("client closed ...")
        # manually kill pbs jobs
        manual_kill_jobs()
        logging.info("manually killed jobs ...")
        # cluster.close()
        # logging.info("cluster closed ...")
    except:
        logging.exception("cluster.close failed ...")
        # manually kill pbs jobs
        manual_kill_jobs()

    logging.info("... done")


def manual_kill_jobs():
    """manually kill dask pbs jobs"""

    import subprocess, getpass

    #
    username = getpass.getuser()
    #
    bashCommand = "qstat"
    try:
        output = subprocess.check_output(
            bashCommand, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    #
    for line in output.splitlines():
        lined = line.decode("UTF-8")
        if username in lined and "dask" in lined:
            pid = lined.split(".")[0]
            bashCommand = "qdel " + str(pid)
            logging.info(" " + bashCommand)
            try:
                boutput = subprocess.check_output(bashCommand, shell=True)
            except subprocess.CalledProcessError as e:
                # print(e.output.decode())
                pass


def dask_compute_batch(computations, client, batch_size=None):
    """breaks down a list of computations into batches"""
    # compute batch size according to number of workers
    if batch_size is None:
        # batch_size = len(client.scheduler_info()["workers"])
        batch_size = sum(list(client.nthreads().values()))
    # find batch indices
    total_range = range(len(computations))
    splits = max(1, np.ceil(len(total_range) / batch_size))
    batches = np.array_split(total_range, splits)
    # launch computations
    outputs = []
    for b in batches:
        logging.info("batches: " + str(b) + " / " + str(total_range))
        out = compute(*computations[slice(b[0], b[-1] + 1)])
        outputs.append(out)

        # try to manually clean up memory
        # https://coiled.io/blog/tackling-unmanaged-memory-with-dask/
        client.run(gc.collect)
        client.run(trim_memory)  # should not be done systematically

    return sum(outputs, ())


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


# ---------------------------------- core of the job to be done ----------------------------------




def run_coloc_swot_250():
    """main execution code"""

    ggrad_variables = ['cvl_mean_dynamic_topography_cnes_cls_22',
                 'cvl_mean_sea_surface_cnes_22_hybrid',
                 'cvl_ocean_tide_fes_2022',
                 'cvl_ssha_reference',
                 'duacs_ssha_karin_2_calibrated',
                 'duacs_ssha_karin_2_filtered',]

    variables = ['duacs_relative_vorticity',
                    'duacs_speed_meridional',
                    'duacs_speed_meridional_abs',
                    'duacs_speed_zonal',
                    'duacs_speed_zonal_abs',
                   'cvl_swh_model', 
                   ]
    #SWOT
    dfs = browse_swot_250().reset_index()
    
    #drifters
    dr = xr.open_dataset(os.path.join(drifters_dir,'L2', 'all_med_variational_10min_v0.nc'))
    dr_key = dr[['cruise_id', 'drifter_type']]
    dr = dr.where(dr.gap_mask==1)

    # select drifters under the swath
    def sel_drifters(swath, cycle):
        dfs_ = dfs.where((dfs.pass_number==swath)&(dfs.cycle_number==cycle)).dropna()
        dr_ = dr.sel(datetime = slice(pd.to_datetime(dfs_.start_time_cut).values[0], pd.to_datetime(dfs_.end_time_cut).values[0]))
        dfr_ = dr_.to_dataframe().reset_index().dropna()
        dfr_['time_to_swot']=(dfr_.datetime-dfs_.time.values[0]).abs()
        
        dfrs_=pd.DataFrame()
        dfrs_['time_to_swot_min'] = dfr_.groupby('drifter_id').time_to_swot.min()
        dfrs_['time_to_swot_max'] = dfr_.groupby('drifter_id').time_to_swot.max()
        dfrs_['point_number'] = dfr_.groupby('drifter_id').datetime.count()
        dfrs_['cycle_number'] = int(dfs_.cycle_number.values[0])
        dfrs_['pass_number'] = int(dfs_.pass_number.values[0])
        return dfs_, dfr_, dfrs_.reset_index()
    
    D = []
    for swath in [3,16]:
        for cycle in dfs.where(dfs.pass_number==swath).dropna().cycle_number:
            dfs_, dfr_, dfrs_ = sel_drifters(swath, cycle)
            dfr_['cycle_number'] = int(dfs_.cycle_number.values[0])
            dfr_['pass_number'] = int(dfs_.pass_number.values[0])
            D.append(dfr_.loc[dfr_.groupby(['drifter_id']).time_to_swot.idxmin()])
    df = pd.concat(D).set_index('pass_number').reset_index()
    df['row_number'] = np.arange(len(df))

    #meta
    df0_ = df.iloc[0:2]
    df_meta = partition_swot_interp_grad(df0_)

    #interp and ggrad
    for i in range(0, len(df), 1000):
        df_ = df.iloc[i:max(i+1000, len(df))]
        from dask.dataframe import from_pandas
        ddf_ = from_pandas(df_, npartitions=50)
        ddf_done = ddf_.map_partitions(partition_compute_grad_to_df, meta=df_meta)
        if i==0:
            ddf_done.to_parquet(os.path.join(zarr_dir, 'DATA_MED_COLOC.parquet'), overwrite=True)
        else : 
            ddf_done.to_parquet(os.path.join(zarr_dir, 'DATA_MED_COLOC.parquet'), append=True)
        logging.info(f"process and store {i+1000}/{len(df)}")

        # close dask
    close_dask(cluster, client)

    logging.info("- all done")
 
    

# ---------------------------------- execution ----------------------------------


if __name__ == "__main__":

    ## step0: setup logging

    # to std output
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # to file
    logging.basicConfig(
        filename="distributed.log",
        level=logging.INFO,
        # level=logging.DEBUG,
    )
    # level order is: DEBUG, INFO, WARNING, ERROR
    # encoding='utf-8', # available only in latests python versions

    # dask memory logging:
    # https://blog.dask.org/2021/03/11/dask_memory_usage
    # log dask summary?

    # spin up cluster
    logging.info("start spinning up dask cluster, jobs={}".format(dask_jobs))
    cluster, client = spin_up_cluster(
        "distributed",
        jobs=dask_jobs,
        fraction=0.9,
        walltime="04:00:00",
        **jobqueuekw,
    )
    ssh_command, dashboard_port = dashboard_ssh_forward(client)
    logging.info("dashboard via ssh: " + ssh_command)
    logging.info(f"open browser at address of the type: http://localhost:{dashboard_port}")
    run_coloc_swot_250()
