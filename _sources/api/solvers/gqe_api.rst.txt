.. function:: gqe(cost, pool, config=None, **kwargs)

    Run the Generative Quantum Eigensolver algorithm.
    
    GQE uses a transformer model to learn which quantum operators from a pool
    should be applied to minimize a given cost function. Python-only implementation.

    The neural network uses `PyTorch <https://pytorch.org/>`_. If the installed PyTorch build
    does not ship CUDA kernels for your GPU, use a wheel with **CUDA 12.8 or newer** that
    supports your architecture (`installation options <https://pytorch.org/get-started/locally/>`_).
    If the driver exposes a GPU but PyTorch cannot execute CUDA kernels on it, ``gqe()``
    writes an error to standard error and exits the process with a non-zero status.

    The GQE implementation in CUDA-Q Solvers is based on this paper: `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_.
    
    :param cost: Cost function that evaluates operator sequences
    :param pool: List of quantum operators to select from
    :param config: Optional configuration object. If None, uses kwargs to override defaults
    :param kwargs: Optional keyword arguments to override default configuration. The following
        special arguments are supported:
        
        - `model`: Can pass in an already constructed transformer
        - `optimizer`: Can pass in an already constructed optimizer
        
        Additionally, any default config parameter can be overridden via kwargs if no
        config object is provided, for example:
        
        - `max_iters`: Overrides cfg.max_iters for total number of epochs to run
        - `energy_offset`: Overrides cfg.energy_offset for offset to add to expectation value
    
    :returns: tuple: Minimum energy found, corresponding operator indices


.. function:: cudaq_solvers.gqe_algorithm.gqe.get_default_config()

    Create a default configuration for GQE.
    
    :returns: Default configuration for GQE with the following parameters:

        - `num_samples` (int): Number of circuits to generate during each epoch/batch. Default=5
        - `max_iters` (int): Number of epochs to run. Default=100
        - `ngates` (int): Number of gates that make up each generated circuit. Default=20
        - `seed` (int): Random seed. Default=3047
        - `lr` (float): Learning rate used by the optimizer. Default=5e-7
        - `energy_offset` (float): Offset added to expectation value of the circuit (Energy) for numerical 
            stability, see `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_ Sec. 3. Default=0.0
        - `grad_norm_clip` (float): max_norm for clipping gradients, see `Lightning docs <https://lightning.ai/docs/fabric/stable/api/fabric_methods.html#clip-gradients>`_. Default=1.0
        - `temperature` (float): Starting inverse temperature β as described in `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_ 
            Sec. 2.2. Default=5.0
        - `del_temperature` (float): Temperature increase after each epoch. Default=0.05
        - `resid_pdrop` (float): The dropout probability for all fully connected layers in the embeddings, 
            encoder, and pooler, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        - `embd_pdrop` (float): The dropout ratio for the embeddings, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        - `attn_pdrop` (float): The dropout ratio for the attention, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        - `small` (bool): Uses a small transformer (6 hidden layers and 6 attention heads as opposed to 
            the default transformer of 12 of each). Default=False
        - `use_fabric_logging` (bool): Whether to enable fabric logging. Default=False
        - `fabric_logger` (object): Fabric logger to use for logging. If None, no logging will be done. Default=None
        - `save_trajectory` (bool): Whether to save the trajectory data to a file. Default=False
        - `trajectory_file_path` (str): Path to save the trajectory data file. Default="gqe_logs/gqe_trajectory.json"
        - `verbose` (bool): Enable verbose output to the console. Output includes the epoch, loss, 
            model.train_step time, and minimum energy. Default=False