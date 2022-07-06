.. highlight:: shell

######################################
Overview
######################################

******************************************************
Condor & Docker Quickstart Guide
******************************************************

This user guide includes an introduction to Condor and Docker and details on using them to manage machine learning jobs within CVIB.
Each section of this document contains an explanation of key components of Docker and Condor and examples of implementing each.
The final section of the document is an example that takes you through creating a Dockerfile and submitting a job to Condor.

**Steps at-a-glance to submit a job to condor:**

1. Get a Docker image from the `public registry <https://www.tensorflow.org/install/docker>`_ or make your own :doc:`Dockerfile<docker>`.
2. If you made your own Dockerfile, build a docker image from the Dockerfile using ``docker build``.
3. Make a :doc:`condor submit file<condor>`.
4. Go to the directory your condor submit file is and use ``condor_submit submit.condor`` to submit the job.
