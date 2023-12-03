 # <p align="center"> Generate Next Texture at Shoulders of Giants:  Stable Diffusion Model using Global Information </p>

 #####  <p align="center"> Cheolhwan Kim, Jinho Park | CS479-2023</p>

## Install

- System requirement: Ubuntu22.04
- Tested GPUs: RTX3090
- CUDA: 12.2
- gcc version: 11.4.0
- Python 3.8.18
- Single GPU

Before install requirements, you should change our gcc source code, and after you install everything, you can freely reverse this process.

in /usr/include/c++/11/bits/std_function.h line 436, 531:
```c
// noexcept(_Handler<_Functor>::template _S_nothrow_init<_Functor>())
```

Use python version 3.8.18:
  ```bash
  conda create -n {env_name} python==3.8.18
  conda activate {env_name}
  ```

Use the file requirements.txt to install all packages one by one. It may fail since the complexity of some packages.
  ```bash
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```

After the successful deployment of the environment, clone the repository of Fantasia3D and get started.
```bash
git clone https://github.com/Gorilla-Lab-SCUT/Fantasia3D.git
cd Fantasia3D
```

## Start
All the results in the paper were generated using 8 3090 GPUs. We cannot guarantee that fewer than 8 GPUs can achieve the same effect.
- zero-shot generation
```bash
# Single GPU training (Only test on the frog). 
# Appearance modeling. It takes about 15 minutes on 3090 GPU.
python3  train.py --config configs/Frog.json
```

## Acknowledgement
- [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D)