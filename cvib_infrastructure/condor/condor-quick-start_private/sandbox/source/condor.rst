.. highlight:: shell

######################################
Condor
######################################

Condor accepts tasks, referred to as jobs, to a queue and manages them by scheduling them to run on avilable compute resources (e.g. GPU clusters). 
Condor assigns units for individual machines called slots, in which a single or multiple jobs are assigned to.
Condor will dynamically create and organize new slots as resources become available for new jobs. To a submit a job using Condor, 
a condor submit file that contains commands to run jobs is needed.

******************************************************
Condor Submit File
******************************************************

Main parameters in a ``submit.condor`` file:

* ``universe``: specify which Condor universe to use when running the job.

* ``docker_image``: specify which docker image to use for running the scriptts

* ``executable``: the shell script used to run a file containing the command line inputs used to run your script that becomes the job. Executables are files saved with **.sh** extension.

* ``arguments``: list of arguments to be supplied to the executable as part of the command line.

* ``requirements``: request resources (e.g. GPU cluster) needed to process job.

* ``log``: creates a job event file.

* ``output``: creates standard output file. 

* ``error``: creates standard error file that captures any error messages. 

* ``tranfer_input_files``: list of all files/directories to be transferred into the working directory for the job.

* ``request_cpus``, ``request_gpus``, ``request_memory``, ``request_disk``: requests needed CPU, GPU, memory and disk space for job. If the requested resources are insufficient for the job, it may cause problems for the user and jobs might be put on hold by condor. If the requested resources are much greater than needed, jobs will match to fewer slots than they could and may block other jobs.

* ``queue``: specify the amount of times to repeat job submission for a set of arguments (defaults to 1).

An example of a ``submit.condor`` file:

    .. code-block:: bash

        universe = docker
        # since universe id docker, specify image from CVIB registry
        docker_image = registry.cvib.ucla.edu/{{username}}:condor-quick-start

        executable = run.sh
        should_transfer_files = YES # transfer files to and from the remote machine where the job runs
        transfer_input_files = run.sh

        requirements = (Machine  == "REDLRADADM14958.ad.medctr.ucla.edu" ||  Machine  == "REDLRADADM14959.ad.medctr.ucla.edu" )

        when_to_transfer_output = ON_EXIT # ON_EXIT transfers job's output files back to the machine when the job completes and exits automatically

        # For logging meta data, use $(process) for Numbered files
        output = log/job.$(cluster).$(process).out
        error = log/job.$(cluster).$(process).err
        log = log/job.$(cluster).$(process).log

        # prior to submitting jobs, if you specified a log directory in the submit script,
        # remember to create the directory first, or job will forever be at idle

        request_cpus = 1
        request_gpus = 1
        request_memory = 1GB
        request_disk = 500MB

        arguments = "hello world"
        queue


Refer to the :doc:`example <example>` section of the documentation to use a Condor submit file for training a CNN.

******************************************************
Useful Commands
******************************************************

Shown below are some commonly used commands to view a job's staus. Screenshots are included below some commands to 
show what sort of output you should see from each one.

* ``condor_submit <submit.condor file>``: submits job(s) specified in the submit file (executable)

* ``condor_q``: monitor/view status of submitted jobs

    .. image:: /screenshots/condor_q_output.png

* ``condor_q -analyze``: check detailed status of job and view how many resources satisfy job requirements

    .. image:: /screenshots/analyze_output.png

* ``condor_q --better-analyze``: shows a more detailed version of ``condor_q`` and ``condor_q -analyze``, useful for diagnosing why jobs are idle

* ``condor_q -nobatch``: displays individual job details

    .. image:: /screenshots/nobatch_output.png

* ``condor_status``: display status of condor resource pool

    .. image:: /screenshots/status_output.png

******************************************************
Resolving Issues
******************************************************
         
Condor may experience Logistical errors and must *hold* a job so that it can be fixed by the user. 

A job on hold is interupted but remains in queue on *"H"* state until it is removed or fixed and released.
Common hold reasons include:

* Job cannot be matched with a machine.
* Incorrect path to fields that need to be transferred.
* Poorly formatted executables.
* Job uses more memory or disk than requested (in condor_submit file).
* Job runs longer than allowed (72-hour default in CHTC pool).
* Admin has to put job on hold.

Refer to the job log, error, and output files for troubleshooting details.

There are a couple ways to view log files for the reason of the held job:

    .. code-block:: bash

        condor_q -hold <Job.ID>
        condor_q -hold -af HoldReason

If the issue requires a resubmission, it can be removed from the queue by:

    .. code-block:: bash

        condor_rm <Job.ID>       # removes job by Job ID
        condor_rm <Cluster.ID>   # removes job by Cluster ID
        condor_rm <Username>     # removes job by username
