1、Our code is implemented based on the diffusers library, so you will need to install the relevant dependencies, including diffusers, torch, and transformers.
2、Download the pre-trained weights for SDv1.4 from the Hugging Face website: https://huggingface.co/CompVis/stable-diffusion-v1-4 and place them in the appropriate location, or allow the code to automatically download them during execution.

3、train step
Navigate to the ptuner directory and run `python train_ptuner.py`. 
The trained direction vector will be saved in the models directory. 
You can modify various parameters in the training script as needed.

4、generation step
Run `python generate.py`. 
You can adjust the parameters within the script as needed.