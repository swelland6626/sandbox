######################################
Example
######################################

Example showing how to use docker and condor to submit a job.

For this example, you will train a simple CNN to do classification of images 
from the mnist dataset with the model shown below. Put this code into a file and save it as **mnnist_ex.py**.

    .. code-block:: bash

        # Example taken from https://keras.io/examples/vision/mnist_convnet/

        # setup
        import numpy as np
        from tensorflow import keras
        from tensorflow.keras import layers


        # prepare the data
        # Model / data parameters
        num_classes = 10
        input_shape = (28, 28, 1)

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        print("x_train shape:", x_train.shape)
        print(x_train.shape[0], "train samples")
        print(x_test.shape[0], "test samples")


        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


        # build the model
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.summary()


        # train the model
        batch_size = 128
        epochs = 10

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


        # evaluate the trained model
        # batch_size = 128
        # epochs = 15

        # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


Next you need to make a Dockerfile so the image knows what dependencies are needed:

    .. code-block:: bash

        FROM tensorflow/tensorflow:2.6.0-gpu-jupyter

        RUN python -m pip install --upgrade pip

        COPY mnist_ex.py .


Then you need to build a docker image and put it in the CVIB registry. From a terminal:

    .. code-block:: bash

        # From a temrinal, cd into your sandbox directory and login to the CVIB docker registry
        docker login registry.cvib.ucla.edu
        
        # Build a docker image (you can use the Dockerfile you created in the docker example):
        docker build -t condor-quick-start .

        # Now export it 
        export REMOTE_URL=registry.cvib.ucla.edu/$USER:condor-quick-start

        # Tag it
        docker tag condor-quick-start $REMOTE_URL

        # Push it
        docker push $REMOTE_URL


Now that the docker image is built, you can use it to launch containers and run scripts.
Create the condor executable to run the model you just made with the code below saved as **run.sh**.

    .. code-block:: bash

        #!/bin/bash
        echo $@

        # change path to your username
        cd /cvib2/apps/personal/{your-username}/sandbox
        CUDA_VISIBLE_DEVICES=0 python mnist_ex.py


Finally, you need to make a condor submit file.

    .. code-block:: bash

        # condor submit file here
        # path to example in git: /condor-quick-start/docker_example/mnist_ex.py

        # path in condor submit file needs to be /sandbox/mnist_ex.py

        universe = docker

        # specify the image from CVIB registry since the universe is is docker
        docker_image = registry.cvib.ucla.edu/{{username}}:condor-quick-start

        executable = run.sh

        # transfer files to and from the remote machine where the job runs
        should_transfer_files = YES 
        transfer_input_files = run.sh

        # tells condor which GPU cluster(s) you want to run on
        requirements = (Machine  == "REDLRADADM14958.ad.medctr.ucla.edu" ||  Machine  == "REDLRADADM14959.ad.medctr.ucla.edu" )

        # ON_EXIT transfers job's output files back to the machine when the job completes and exits automatically
        when_to_transfer_output = ON_EXIT

        # For logging meta data, use $(process) for Numbered files
        output = joblog/job.$(cluster).$(process).out
        error = joblog/job.$(cluster).$(process).err
        log = joblog/job.$(cluster).$(process).log

        # prior to submitting jobs, if you specified a log directory in the submit script,
        # remember to create the directory first, or job will forever be at idle

        request_cpus = 1
        request_gpus = 1
        request_memory = 1GB
        request_disk = 500MB

        arguments = "hello world"
        queue


Since the docker image has been built and you have a condor submit file, now you can submit the job!

    .. code-block:: bash

        # Submit the job to condor
        condor_submit submit.condor


        # to view job status
        condor_q


You can view the resource the job is using by 
going `here <http://radcondor.cvib.ucla.edu:48109/>`_ and finding your username (this model doesn't actually use a GPU but normally this is how you view resource usage). 


**Tip**: In this case, you don't need to launch the container to run the job like you did in the example in the docker section. 
This is because the container is launched by the submit file based on the image specified with ``docker_image =``. If you wanted 
to start the job locally instead of using condor, you would need to launch the container and then run the model script like a normal python file:

    .. code-block:: bash

        # in a terminal launch a docker container based on the image you built
        docker run -it -v $PWD:/workdir -w /workdir -v /radraid:/radraid -u $(id -u):$(id -g) condor-quick-start bash

        # this is a lightweight job that doesn't need a GPU so just run with this:
        python mnist_ex.py


You can exit the docker container using ``ctrl+d`` or ``exit``.
