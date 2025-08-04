import streamlit as st, torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from PIL import Image
from model import FashionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device); model.eval()

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.set_page_config(page_title="Fashion Classifier", layout="centered")
with open("style.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("👗 Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image or use a random test image")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = FashionMNIST(root="./data", train=False, download=True, transform=transform)

if st.button("🎲 Random Test Image"):
    import random
    idx = random.randint(0, len(test_set) - 1)
    image, label = test_set[idx]
    out = model(image.unsqueeze(0).to(device))
    pred = out.argmax(1).item()
    st.image(image.squeeze().numpy(), caption=f"True: {classes[label]} | Predicted: {classes[pred]}", width=200)

file = st.file_uploader("Upload image", type=["png", "jpg"])
if file:
    img = Image.open(file).convert("L").resize((28, 28))
    st.image(img, caption="Uploaded Image", width=200)
    tensor = transform(img).unsqueeze(0).to(device)
    out = model(tensor)
    pred = out.argmax(1).item()
    st.write(f"### 🧠 Prediction: **{classes[pred]}**")
