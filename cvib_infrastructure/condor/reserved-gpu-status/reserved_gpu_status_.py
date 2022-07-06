'''
Run a script to output either a print statement or a status text file to say which GPUs are being reserved in Condor.

Process: 
    Search all jobs in Condor for each GPU core and see if it is being used. 
    If found, list the machine, core, job owner, and job id.

subprocess = "condor_q -run -l --global --all-users"

Use this on every machine to get a lookup table of the GPU cores
gpu_table = "nvidia-smi -L - "

Use /cvib2/apps/personal/swelland/sandbox/test-area/condor/reserved-gpu-status/status.sh to run subprocess for updating file and searching contents

'''

import subprocess
import io


gpu_dict = {'AssignedGPUs = "GPU-9e145a8a"': {"core": 0, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-83a62fe3"': {"core": 1, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-f1cdaf01"': {"core": 2, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-4155d4c1"': {"core": 3, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-f47e8274"': {"core": 4, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-c543f5d8"': {"core": 5, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-2ce9448b"': {"core": 6, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-20a7abae"': {"core": 7, "machine": "REDLRADADM23589"}, 
            'AssignedGPUs = "GPU-cb3e5aa8"': {"core": 0, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-7da3e820"': {"core": 1, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-8c4bbdd7"': {"core": 2, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-113d8fe6"': {"core": 3, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-3bfb8a1f"': {"core": 4, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-340484a8"': {"core": 5, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-76ce9bdd"': {"core": 6, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-0f92030d"': {"core": 7, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-99206ef3"': {"core": 8, "machine": "REDLRADADM14958"}, 
            'AssignedGPUs = "GPU-814f208a"': {"core": 0, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-aa89f48d"': {"core": 1, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-41bf3322"': {"core": 2, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-bc66d487"': {"core": 3, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-cc612718"': {"core": 4, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-d4aea729"': {"core": 5, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-87fa5dae"': {"core": 6, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-5ec9f7f8"': {"core": 7, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-36a15eb0"': {"core": 8, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-710d6c7f"': {"core": 9, "machine": "REDLRADADM14959"}, 
            'AssignedGPUs = "GPU-8c8f79d1"': {"core": 0, "machine": "REDLRADADM23710"}, 
            'AssignedGPUs = "GPU-8c5473f0"': {"core": 1, "machine": "REDLRADADM23710"}, 
            'AssignedGPUs = "GPU-275dd631"': {"core": 2, "machine": "REDLRADADM23710"}, 
            'AssignedGPUs = "GPU-774ab04c"': {"core": 3, "machine": "REDLRADADM23710"}, 
            'AssignedGPUs = "GPU-a10cce8e"': {"core": 0, "machine": "REDLRADADM23620"}, 
            'AssignedGPUs = "GPU-c3eeda07"': {"core": 1, "machine": "REDLRADADM23620"}, 
            'AssignedGPUs = "GPU-be62e17e"': {"core": 2, "machine": "REDLRADADM23620"}, 
            'AssignedGPUs = "GPU-15b40d40"': {"core": 3, "machine": "REDLRADADM23620"}, 
            'AssignedGPUs = "GPU-d451a4be"': {"core": 0, "machine": "REDLRADADM23621"}, 
            'AssignedGPUs = "GPU-ad6186a5"': {"core": 1, "machine": "REDLRADADM23621"}, 
            'AssignedGPUs = "GPU-02ab14df"': {"core": 2, "machine": "REDLRADADM23621"}, 
            'AssignedGPUs = "GPU-3f0c79f5"': {"core": 3, "machine": "REDLRADADM23621"}, 
            'AssignedGPUs = "GPU-ab0449b3"': {"core": 0, "machine": "redlradbei05920"}, 
            'AssignedGPUs = "GPU-1d521ac3"': {"core": 1, "machine": "redlradbei05920"}, 
            'AssignedGPUs = "GPU-a9c609b7"': {"core": 2, "machine": "redlradbei05920"}, 
            'AssignedGPUs = "GPU-04cbc1a3"': {"core": 3, "machine": "redlradbei05920"}, 
            'AssignedGPUs = "GPU-8645ab88"': {"core": 0, "machine": "REDWRADMMC23199"}, 
            'AssignedGPUs = "GPU-839a41dd"': {"core": 1, "machine": "REDWRADMMC23199"}, 
            'AssignedGPUs = "GPU-7e6e8bcc"': {"core": 2, "machine": "REDWRADMMC23199"}, 
            'AssignedGPUs = "GPU-1a8cec36"': {"core": 3, "machine": "REDWRADMMC23199"}, 
            'AssignedGPUs = "GPU-39e42370"': {"core": 0, "machine": "REDWRADADM23712"}, 
            'AssignedGPUs = "GPU-6bd38320"': {"core": 1, "machine": "REDWRADADM23712"}, 
            'AssignedGPUs = "GPU-00cbe557"': {"core": 2, "machine": "REDWRADADM23712"}, 
            'AssignedGPUs = "GPU-13c10646"': {"core": 0, "machine": "REDLRADADM11249"}, 
            'AssignedGPUs = "GPU-8c662d68"': {"core": 1, "machine": "REDLRADADM11249"}, 
            'AssignedGPUs = "GPU-cb1f2d3d"': {"core": 2, "machine": "REDLRADADM11249"}, 
            'AssignedGPUs = "GPU-32885b24"': {"core": 3, "machine": "REDLRADADM11249"}, 
            }


gpu_id = "AssignedGPUs"
owner = "Owner ="
# global_job_id = "GlobalJobId" # MWW 061422
machine_matched = "LastMatchName0" # MWW 061422
remote_host = "RemoteHost" # MWW 061422
last_remote_host = "LastRemoteHost"
cluster_id = "ClusterId =" # MWW 061422
autocluster_id = "AutoClusterId ="


# path = '/cvib2/apps/personal/wasil/trash/condor.txt'
# # path = '/cvib2/apps/personal/swelland/sandbox/test-area/condor/reserved-gpu-status/condor_static.txt'

# with open(path, 'r') as f:
#     contents = f.read()

# jobs = contents.split("\n\n")   # makes new list where each element contains all details for a job

sp = subprocess.Popen(['bash', '-c', "condor_q -run -l --global --all-users"], stdout=subprocess.PIPE)
condorout = io.TextIOWrapper(sp.stdout, encoding="utf-8")

lines = condorout.read()
jobs = lines.split("\n\n")

dict_list = []

for job in jobs:
    if gpu_id in job:
        gpu_job = job.split("\n")
        my_dict = {}

        for i in gpu_job:
            if gpu_id in i:
                current_machine = gpu_dict[i]["machine"]
                core = gpu_dict[i]["core"]


                my_dict["machine"] = current_machine
                my_dict["core"] = core

            elif remote_host in i and not last_remote_host in i:
                if current_machine in i:
                    req_gpu = True
                    my_dict["req_gpu"] = req_gpu
            
                else:
                    req_gpu = False
                    my_dict["req_gpu"] = req_gpu
            
            elif owner in i:
                my_dict["owner"] = i

            elif cluster_id in i and autocluster_id not in i:
                my_dict["cluster_id"] = i

        dict_list.append(my_dict)


job_list = []
no_gpu_list = []
idx = 0

for i in dict_list:
    if dict_list[idx]["req_gpu"]:
        job_list.append(i)
        idx += 1
    else:
        no_gpu_list.append(i)
        idx += 1


print()
gpus_reserved = len(job_list)
no_gpus = len(no_gpu_list)
print(f"GPUs reserved: {gpus_reserved}")
print(f"{no_gpus} jobs originally assigned a GPU but no longer need one.")
print()

# sorts by machine and core in ascending order
# cores_sorted = sorted(job_list, key=lambda x: x["core"])
# names_sorted = sorted(cores_sorted, key=lambda x: x["machine"])
gpus_sorted = sorted(job_list, key=lambda x: (x["machine"], x["core"]))


idx = 0
for i in gpus_sorted:
    core = gpus_sorted[idx]["core"]
    machine = gpus_sorted[idx]["machine"]
    owner = gpus_sorted[idx]["owner"]
    cluster_id = gpus_sorted[idx]["cluster_id"]

    print(f"[{core}], {machine}, {owner}, {cluster_id}")
    idx += 1

print()