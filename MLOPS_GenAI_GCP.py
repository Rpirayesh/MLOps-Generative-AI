from google.colab import files
uploaded = files.upload()
  
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/Your ENV.json"

!pip install git+https://github.com/openai/shap-e.git
!pip install google-cloud-pubsub
!pip install trimesh
import trimesh

import shap_e


import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xm = load_model('transmitter', device=device)
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))


!pip install google-cloud-pubsub


from google.cloud import pubsub_v1
import time

# Create a subscriber client
subscriber = pubsub_v1.SubscriberClient()

# Define your Pub/Sub subscription
subscription_path = subscriber.subscription_path("Your Path", "unity-subscription")

# Define a callback function to handle incoming messages
def callback(message):
    global prompt
    prompt = message.data.decode('utf-8')
    print(f"Received message: {prompt}")
    message.ack()
    global message_received
    message_received = True

message_received = False

# Subscribe to the topic
subscriber.subscribe(subscription_path, callback=callback)

# The subscriber is non-blocking, so we must keep the script from exiting to allow it to process messages
# while not message_received:
#     time.sleep(5)  # Wait for a bit to allow messages to be processed and the prompt to be updated

# Use the received message as the prompt
batch_size = 1
guidance_scale = 10.0

latents = sample_latents(
    batch_size=batch_size,
    model=model,
    diffusion=diffusion,
    guidance_scale=guidance_scale,
    model_kwargs=dict(texts=[prompt] * batch_size),
    progress=True,
    clip_denoised=True,
    use_fp16=True,
    use_karras=True,
    karras_steps=64,
    sigma_min=1e-3,
    sigma_max=160,
    s_churn=0,
)

render_mode = 'nerf' # you can change this to 'stf'
size = 64 # this is the size of the renders; higher values take longer to render.

cameras = create_pan_cameras(size, device)
for i, latent in enumerate(latents):
    images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
    display(gif_widget(images))

# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh

for i, latent in enumerate(latents):
    t = decode_latent_mesh(xm, latent).tri_mesh()
    with open(f'example_mesh_{i}.ply', 'wb') as f:
        t.write_ply(f)
    with open(f'example_mesh_{i}.obj', 'w') as f:
        t.write_obj(f)
    # with open(f'example_mesh_{i}.mtl', 'w') as f:
    #    t.write_mtl(f)


# Load your .ply file
mesh = trimesh.load_mesh('example_mesh_0.ply')

# Export the mesh as .obj
mesh.export('output_3d.glb')

from google.cloud import storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    # Make the blob publicly readable
    blob.make_public()

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))
    print("The public url is: {}".format(blob.public_url))
    # Upload the .obj files to Google Cloud Storage
for i in range(len(latents)):
    # obj_filename = f'example_mesh_{i}.glb'
    obj_filename = 'output_3d.glb'
    upload_blob('reza_vr', obj_filename, f'3DModels/{obj_filename}')