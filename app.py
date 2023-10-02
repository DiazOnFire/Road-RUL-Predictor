import streamlit as st 
import pickle
import numpy as np
import PIL
from PIL import Image
import ultralytics
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="RUL Predictor", 
    page_icon="🤖",    
    layout="wide",      
    initial_sidebar_state="expanded"    
)



page_bg_img ="""
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://wallpapercave.com/wp/wp9116523.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
    background-color:rgba(0,0,0,0);
    
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


with st.sidebar:
    st.header("Choose an Image or Upload one")
    
    default_imgs=os.listdir("defaults")
    selected_image=st.selectbox("Select one from here",["Choose an image"]+default_imgs)
   
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    
    
    
    


def output_fetch(results):
    
    results = [result.cpu() for result in results]
    lis=[]
  
    results = [result.numpy() for result in results]
   
    boxes = results[0].boxes
    for box in boxes:
        type_of_damage=results[0].names[box.cls[0]]
        dpi=96
        px_to_mm=25.4/dpi
      
        bbox = box.xyxy[0]
    
        image_width = results[0].orig_img.shape[1]*px_to_mm
        image_height = results[0].orig_img.shape[0]*px_to_mm
    
        x1 = bbox[0] * image_width
        y1 = bbox[1] * image_height
        x2 = bbox[2] * image_width
        y2 = bbox[3] * image_height
    
        width = (x2 - x1)*px_to_mm
        height = (y2 - y1)*px_to_mm
    
        width=width*px_to_mm
        height=height*px_to_mm
    
        area = width * height
        area=area*(px_to_mm**2)
    
        image_area = image_width * image_height
    
        relative_area = (area / image_area)
        lis.append([type_of_damage,((width+height)/2),relative_area])
        
        
    return lis
        

def modify(lis):
    finals=[]
    for dmges in lis:
    #Get crack_type
        if dmges[0] in ['D00','D01','D10','D11']:
            crack_type="linear"
        elif dmges[0]=='D20':
            crack_type="alligator"
        else:
            crack_type="other"
         
        #Get severity rating    
        if dmges[1]<=500:
            severity_rating=1
        elif dmges[1]>500 and dmges[1]<=800:
            severity_rating=2
        elif dmges[1]>800 and dmges[1]<=1200:
            severity_rating=3
        elif dmges[1]>1200 and dmges[1]<=1500:
            severity_rating=4
        else:
            severity_rating=5
        
        #Get density_rating
        if dmges[2]<=3:
            density_rating=1
        elif dmges[2]>3 and dmges[2]<=6:
            density_rating=2
        elif dmges[2]>6 and dmges[2]<=10:
            density_rating=3
        elif dmges[2]>10 and dmges[2]<=14:
            density_rating=4
        else:
            density_rating=5
        
        finals.append([crack_type,severity_rating,density_rating])
    return finals
        

def PCI_Calc(finals):
  dmi=0
  
  for f in finals:
    weight = {
      "linear": 2.7,
      "alligator": 2.2,
      "other": 1.5,
      }.get(f[0])
    dmi+=(f[1]+f[2])*weight


  pci=100-dmi
  return pci


    
st.title("Prediction Of Remaining Useful Life of Road")
st.write("Select or upload an image and click on Estimate RUL to get results")
col1,col2=st.columns(2)

with col1:
    if selected_image!="Choose an image":
        source_img=os.path.join("defaults",selected_image)   
        uploaded_image=PIL.Image.open(source_img)
        st.image(source_img,caption="Uploaded Image",use_column_width=True)
    elif source_img:
        uploaded_image=PIL.Image.open(source_img)
        st.image(source_img,caption="Uploaded Image",use_column_width=True)
        
try:
    model=YOLO("best.pt")
except Exception as ex:
    st.error(
    f"Unable to load model. Check the specified path")
    st.error(ex)
    
    
    

if st.sidebar.button('Estimate RUL'):
    
    results=model.predict(uploaded_image)
    pci_value=PCI_Calc(modify(output_fetch(results)))
    
        
    boxes=results[0].boxes
    plotted=results[0].plot()[:,:,::-1]
    with col2:
        st.image(plotted,caption="Detected Image",use_column_width=True)
        RUL=4.1872*(np.log(pci_value))-14.117
    
    if pci_value>70:
        quality="Good and does not need improvement, Maintenance is not urgent"
    elif pci_value<70 and pci_value>55:
        quality="Fair but needs maintenace soon,scope for improvement present"
    else:
        quality="Need Maintenance, Poor Quality and less service life"
        
    st.subheader("The pavement condition Index value of the road section given is\t"+str(pci_value))
    st.subheader(quality)
    st.title("The Remaining Useful Life of this road is\t"+  str(round(RUL,2))+"\t Years")
    




