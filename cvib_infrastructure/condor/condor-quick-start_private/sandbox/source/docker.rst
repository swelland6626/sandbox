******************************
Docker
******************************

Docker is a tool for building and using software based on the concept of packaging
code and it's dependencies into a single unit using containers and images.
Images specify everything the container needs to run the scripts inside of it.
Containers are made from images and are the single units that run the code inside of them. 
Images are like snapshots of the contents of the container that get saved for building new 
containers similar to the current one.

The special characteristic of containers is that they don't make
permanent changes to anything outside the container, i.e. to the image. So 
whatever happens inside the container is gone after the container is exited. 


++++++++++++++++++++++++++++++
Dockerfiles
++++++++++++++++++++++++++++++

Docker images are built using Dockerfiles which contain 'layers'. Each instruction 
in a Dockerfile creates a new layer. Common instructions in a 
Dockerfile are described below (convention is for instructions to be uppercase to distinguish them 
from arguments/commands):

*  ``FROM``: Specifies parent image you want to build your image from (the first command in a Dockerfile **must** be ``FROM``; comments and args used in ``FROM`` are an exception).
*  ``PULL``: Adds files from Docker repository.
*  ``RUN``: Executes commands in layers above and commits them for the container built from the next layer.
*  ``COPY``: Copies new files, directories, or remote URLs from ``<src>`` and adds them to the filesystem of the image path ``<dest>``.
*  ``CMD``: Specifies the command to run in the container. There can only be **one** ``CMD`` instruction in a Dockerfile.

See below for more detailed Docker documentation:

* Dockerfile reference:

   * https://docs.docker.com/engine/reference/builder/

* Specific documentation on ``docker build``:

   * https://docs.docker.com/engine/reference/commandline/build/

* Specific documentation on ``docker run``:

   * https://docs.docker.com/engine/reference/run/
   * https://docs.docker.com/engine/reference/commandline/run/

* Dockerfile writing tips:

   * https://docs.docker.com/develop/develop-images/dockerfile_best-practices/


++++++++++++++++++++++++++++++
Dockerfile Examples
++++++++++++++++++++++++++++++

A simple Dockerfile:

   .. code-block:: bash

      # uses tensorflow/tensorflow:2.6.0-gpu-jupyter as the parent image to build the container 
      FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

      # upgrades the pip installer
      RUN python -m pip install --upgrade pip

      # requirements.txt is a file containing dependencies to install inside the container
      # copies requirements.txt into the container, the " ." at the end specifies to copy it to the current directory
      COPY requirements.txt .

      # uses the pip installer to install the contents (dependencies) of requirements
      RUN pip install -r requirements.txt


A more complex Dockerfile:

   .. code-block:: bash

      FROM sphinxdoc/sphinx-latexpdf
      # FROM sphinxdoc/sphinx

      ARG MYPATH=/usr/local
      ARG MYLIBPATH=/usr/lib

      RUN apt-get update && apt-get install -y --no-install-recommends \
            autotools-dev \
            build-essential \
            ca-certificates \
            cmake \
            git \
            wget \
            curl \
            vim
      RUN rm -rf /var/lib/apt/lists/*

      # install miniconda.
      # create and activate python virtual env with desired version
      RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh --no-check-certificate -O ~/miniconda.sh && \
         /bin/bash ~/miniconda.sh -b -p /opt/conda
      RUN /opt/conda/bin/conda create -n py python=3.7.2
      RUN echo "source /opt/conda/bin/activate py" > ~/.bashrc
      ENV PATH /opt/conda/envs/py/bin:$PATH
      RUN /bin/bash -c "source /opt/conda/bin/activate py"

      RUN /bin/bash -c "source /opt/conda/bin/activate py && conda install cython numpy -y && pip install scikit-build && pip install matplotlib"
      RUN /bin/bash -c "source /opt/conda/bin/activate py && conda install -c conda-forge jupyterlab -y"
      RUN /bin/bash -c "source /opt/conda/bin/activate py && conda install -c conda-forge nbsphinx -y"

      RUN pip install sphinx-rtd-theme numpydoc sphinx-copybutton
      # RUN pip install ipywidgets matplotlib medpy opencv-python plotly tabulate
      # RUN pip install tensorflow pandas scikit-image pydicom

      # ARG UNAME=testuser
      # ARG UID=1000
      # ARG GID=1000
      # RUN groupadd -g $GID -o $UNAME
      # RUN useradd -l -m -u $UID -g $GID -o -s /bin/bash $UNAME && \
      #     usermod -aG sudo $UNAME
      # RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
      # USER $UNAME

      # CMD /bin/bash


++++++++++++++++++++++++++++++
Docker Images
++++++++++++++++++++++++++++++

Once the Dockerfile is written and you're ready to use it/run your scripts, the 
Docker image must be built. For CVIB, you will also want to push the 
image to the CVIB registry which is a private hub for saved/committed Docker images. 

For practice, create a directory inside your personal directory called 'sandbox' , cd into it
and make a new file called **hello.py** and write the following code inside:

   .. code-block:: bash

      import sys,os
      import imageio
      import numpy as np

      image_path = sys.argv[1]
      output_path = sys.argv[2]

      os.makedirs(os.path.dirname(os.path.abspath(output_path)),exist_ok=True)

      a = imageio.imread(image_path)

      with open(output_path,'w') as f:
         f.write(str(np.sum(a))+'\n')

Create another file saved as **hello.sh** and write the following code inside:

   .. code-block:: bash

      #!/bin/bash
      export url=$1
      export outputpath=$2

      wget $url -O image.png
      python hello.py image.png $outputpath
      cat $outputpath


Create another file saved as **Dockerfile** and write the following code inside:

   .. code-block:: bash

      FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

      RUN apt-get -yq update;apt-get install -yq vim wget
      RUN pip3 install imageio==2.6.0

      WORKDIR /opt
      COPY hello.py .
      COPY hello.sh .

      COPY requirements.txt .
      RUN pip3 install -r requirements.txt


Finally, run the following commands in a terminal to launch the container you made and test it out.

   .. code-block:: bash
          
      # login to CVIB registry
      docker login registry.cvib.ucla.edu

      # build image
      docker build -t hello-tf .

      # launch the container to test it using interactive mode
      docker run -it --privileged  hello-tf /bin/bash
      /bin/bash hello.sh https://i.stack.imgur.com/Ds5Rc.png results.txt
      exit

      # push the image to the CVIB registry
      docker tag hello-tf:latest registry.cvib.ucla.edu/$USER:hello-tf
      docker push registry.cvib.ucla.edu/$USER:hello-tf

      # launch the container
      docker run -v $PWD:/out registry.cvib.ucla.edu/$USER:hello-tf /bin/bash hello.sh https://i.stack.imgur.com/Ds5Rc.png /out/results.txt
      cat results.txt



++++++++++++++++++++++++++++++
Useful commands
++++++++++++++++++++++++++++++

Some useful terminal commands in case you run into issues where you close a terminal
without exiting the docker container. Doing so will result in an error message that 
says something about the port already being allocated.

* ``docker image ls``: lists all docker containers and their IDs
* ``docker rmi -f container-id``: removes a running docker image
* ``docker exec``: runs a command inside a a running container (similar to ``docker run``)
* ``docker exec -it [container-id] bash``: enters an already running docker


When inside a container, ``exit`` or the keystroke ``ctrl+d`` will exit the docker. 

Use a ``.dockerignore`` file to exclude files from the container build. Usage is 
similar to a ``.gitignore`` file.

