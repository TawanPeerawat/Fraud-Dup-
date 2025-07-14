import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import requests
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import json
import google.generativeai as genai

# กำหนดค่า Streamlit
st.set_page_config(
    page_title="Image Similarity Checker",
    page_icon="🖼️",
    layout="wide"
)

# ฟังก์ชันสำหรับแปลงรูปภาพเป็น base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ฟังก์ชันสำหรับคำนวณความเหมือนด้วย SSIM
def calculate_ssim(img1, img2):
    # แปลงเป็น grayscale
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    
    # ปรับขนาดให้เท่ากัน
    h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (w, h))
    gray2 = cv2.resize(gray2, (w, h))
    
    # คำนวณ SSIM
    similarity_score = ssim(gray1, gray2)
    return similarity_score

# ฟังก์ชันสำหรับแยก features ด้วย ResNet
@st.cache_resource
def load_feature_extractor():
    model = resnet50(pretrained=True)
    model.eval()
    # เอาเฉพาะ feature extractor (ไม่รวม classifier)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    return feature_extractor

def extract_features(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0)
        features = model(image_tensor)
        features = features.flatten().numpy()
    
    return features

# ฟังก์ชันสำหรับคำนวณความเหมือนด้วย Deep Learning
def calculate_deep_similarity(img1, img2, model):
    features1 = extract_features(img1, model)
    features2 = extract_features(img2, model)
    
    # คำนวณ cosine similarity
    similarity = cosine_similarity([features1], [features2])[0][0]
    return similarity

# ฟังก์ชันสำหรับเรียก Gemini API
def get_ai_analysis(image1, image2, ssim_score, deep_score, api_key):
    try:
        # กำหนดค่า Gemini API
        genai.configure(api_key=api_key)
        
        # สร้าง model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # สร้าง prompt
        prompt = f"""
        วิเคราะห์ความเหมือนของภาพทั้งสองนี้อย่างละเอียด:
        
        ผลการวิเคราะห์เบื้องต้น:
        - SSIM Score: {ssim_score:.4f} ({ssim_score*100:.2f}%)
        - Deep Learning Score: {deep_score:.4f} ({deep_score*100:.2f}%)
        
        กรุณาวิเคราะห์และอธิบายผลลัพธ์อย่างละเอียดว่า:
        1. ภาพทั้งสองเหมือนกันหรือไม่ และเหมือนกันกี่เปอร์เซ็นต์ตามการประเมินของคุณ
        2. จุดที่เหมือนกันและแตกต่างกันคืออะไร (สี, รูปร่าง, เนื้อหา, การจัดวาง)
        3. เหตุผลที่ทำให้ได้คะแนน SSIM และ Deep Learning แบบนี้
        4. ข้อจำกัดของการวิเคราะห์และคำแนะนำ
        5. สรุปผลการเปรียบเทียบแบบง่ายๆ
        
        ตอบเป็นภาษาไทยและอธิบายให้เข้าใจง่าย ใช้ markdown format และ emoji เพื่อให้อ่านง่าย
        """
        
        # ส่งภาพและ prompt ไปยัง Gemini
        response = model.generate_content([prompt, image1, image2])
        
        return response.text
        
    except Exception as e:
        # ถ้าเกิดข้อผิดพลาดในการเรียก API ให้ใช้ fallback response
        st.error(f"ข้อผิดพลาดในการเรียก Gemini API: {str(e)}")
        
        # Fallback response
        fallback_response = f"""
        ## ⚠️ การวิเคราะห์ด้วยระบบพื้นฐาน
        
        **ผลลัพธ์โดยรวม:** ภาพทั้งสองมีความเหมือนกัน **{((ssim_score + deep_score) / 2 * 100):.1f}%**
        
        ### 🔍 การวิเคราะห์เชิงโครงสร้าง (SSIM: {ssim_score*100:.1f}%)
        - วิเคราะห์โครงสร้างพื้นฐาน ความสว่าง และความคมชัดของภาพ
        - {"คะแนนสูง แสดงว่าภาพมีโครงสร้างคล้ายกันมาก" if ssim_score > 0.8 else "คะแนนปานกลาง อาจมีความแตกต่างในโครงสร้าง" if ssim_score > 0.5 else "คะแนนต่ำ ภาพมีโครงสร้างแตกต่างกันมาก"}
        
        ### 🧠 การวิเคราะห์เชิงลึก (Deep Learning: {deep_score*100:.1f}%)
        - ใช้ AI วิเคราะห์เนื้อหาและรูปแบบเชิงลึก
        - {"AI ตรวจจับว่าเนื้อหาภาพเหมือนกันมาก" if deep_score > 0.8 else "AI ตรวจจับความเหมือนปานกลาง" if deep_score > 0.5 else "AI ตรวจจับว่าเนื้อหาภาพแตกต่างกัน"}
        
        ### 📊 สรุปผลการเปรียบเทียบ
        - **ความเหมือนโดยรวม:** {((ssim_score + deep_score) / 2 * 100):.1f}%
        - **ความเหมือนด้านโครงสร้าง:** {ssim_score*100:.1f}%
        - **ความเหมือนด้านเนื้อหา:** {deep_score*100:.1f}%
        
        ### 💡 คำแนะนำ
        {"ภาพทั้งสองมีความเหมือนกันสูงมาก น่าจะเป็นภาพเดียวกันหรือภาพที่มีเนื้อหาเหมือนกันมาก" if (ssim_score + deep_score) / 2 > 0.8 else "ภาพมีความเหมือนปานกลาง อาจเป็นภาพเดียวกันที่ผ่านการแก้ไขหรือภาพที่คล้ายกัน" if (ssim_score + deep_score) / 2 > 0.5 else "ภาพทั้งสองแตกต่างกันมาก น่าจะเป็นภาพต่างกัน"}
        
        ---
        *หมายเหตุ: ไม่สามารถเชื่อมต่อกับ Gemini AI ได้ กรุณาตรวจสอบ API Key*
        """
        
        return fallback_response

# Main App
def main():
    st.title("🖼️ Image Similarity Checker with AI Analysis")
    st.markdown("### ตรวจสอบความเหมือนของรูปภาพด้วย AI")
    
    # Load model
    with st.spinner("กำลังโหลด AI Model..."):
        feature_extractor = load_feature_extractor()
    
    # Sidebar
    st.sidebar.header("ตัวเลือก")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "🔑 Gemini API Key",
        type="password",
        help="ใส่ Google AI Studio API Key ของคุณ"
    )
    
    analysis_method = st.sidebar.selectbox(
        "เลือกวิธีการวิเคราะห์",
        ["ทั้งหมด", "SSIM เท่านั้น", "Deep Learning เท่านั้น"]
    )
    
    # วิธีการได้ API Key
    with st.sidebar.expander("📝 วิธีการได้ API Key"):
        st.markdown("""
        1. ไปที่ [Google AI Studio](https://aistudio.google.com/)
        2. คลิก "Get API Key"
        3. สร้าง API Key ใหม่
        4. Copy API Key มาใส่ในช่องด้านบน
        """)
    
    # ตรวจสอบ API Key
    if not api_key:
        st.sidebar.warning("⚠️ กรุณาใส่ Gemini API Key เพื่อใช้งาน AI Analysis")
    else:
        st.sidebar.success("✅ API Key ถูกต้อง")
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ภาพที่ 1")
        uploaded_file1 = st.file_uploader(
            "อัปโหลดภาพแรก",
            type=['png', 'jpg', 'jpeg'],
            key="img1"
        )
        
    with col2:
        st.subheader("ภาพที่ 2")
        uploaded_file2 = st.file_uploader(
            "อัปโหลดภาพที่สอง",
            type=['png', 'jpg', 'jpeg'],
            key="img2"
        )
    
    if uploaded_file1 is not None and uploaded_file2 is not None:
        # แสดงภาพ
        col1, col2 = st.columns(2)
        
        with col1:
            image1 = Image.open(uploaded_file1).convert('RGB')
            st.image(image1, caption="ภาพที่ 1", use_column_width=True)
            
        with col2:
            image2 = Image.open(uploaded_file2).convert('RGB')
            st.image(image2, caption="ภาพที่ 2", use_column_width=True)
        
        # ปุ่มวิเคราะห์
        if st.button("🔍 วิเคราะห์ความเหมือน", type="primary"):
            with st.spinner("กำลังวิเคราะห์..."):
                
                ssim_score = None
                deep_score = None
                
                # คำนวณ SSIM
                if analysis_method in ["ทั้งหมด", "SSIM เท่านั้น"]:
                    ssim_score = calculate_ssim(image1, image2)
                
                # คำนวณ Deep Learning similarity
                if analysis_method in ["ทั้งหมด", "Deep Learning เท่านั้น"]:
                    deep_score = calculate_deep_similarity(image1, image2, feature_extractor)
                
                # แสดงผลลัพธ์
                st.subheader("📊 ผลการวิเคราะห์")
                
                # แสดงคะแนน
                col1, col2, col3 = st.columns(3)
                
                if ssim_score is not None:
                    with col1:
                        st.metric(
                            "SSIM Score",
                            f"{ssim_score:.4f}",
                            f"{ssim_score*100:.2f}%"
                        )
                
                if deep_score is not None:
                    with col2:
                        st.metric(
                            "Deep Learning Score",
                            f"{deep_score:.4f}",
                            f"{deep_score*100:.2f}%"
                        )
                
                if ssim_score is not None and deep_score is not None:
                    overall_score = (ssim_score + deep_score) / 2
                    with col3:
                        st.metric(
                            "Overall Similarity",
                            f"{overall_score:.4f}",
                            f"{overall_score*100:.2f}%"
                        )
                
                # Progress bars
                if ssim_score is not None:
                    st.subheader("📈 Visual Progress")
                    st.progress(ssim_score, text=f"SSIM: {ssim_score*100:.1f}%")
                
                if deep_score is not None:
                    st.progress(deep_score, text=f"Deep Learning: {deep_score*100:.1f}%")
                
                # AI Analysis
                st.subheader("🤖 AI Analysis")
                
                if api_key:
                    with st.spinner("กำลังให้ AI วิเคราะห์ภาพ..."):
                        # เรียก Gemini API analysis
                        ai_analysis = get_ai_analysis(
                            image1, 
                            image2, 
                            ssim_score or 0, 
                            deep_score or 0,
                            api_key
                        )
                        
                        st.markdown(ai_analysis)
                else:
                    st.warning("⚠️ กรุณาใส่ Gemini API Key ในแถบด้านข้างเพื่อใช้งาน AI Analysis")
                
                # เพิ่มข้อมูลทางเทคนิค
                with st.expander("ข้อมูลทางเทคนิค"):
                    st.write("**ข้อมูลภาพที่ 1:**")
                    st.write(f"- ขนาด: {image1.size}")
                    st.write(f"- โหมดสี: {image1.mode}")
                    
                    st.write("**ข้อมูลภาพที่ 2:**")
                    st.write(f"- ขนาด: {image2.size}")
                    st.write(f"- โหมดสี: {image2.mode}")
                    
                    st.write("**วิธีการวิเคราะห์:**")
                    st.write("- SSIM: Structural Similarity Index")
                    st.write("- Deep Learning: ResNet50 feature extraction + Cosine similarity")

if __name__ == "__main__":
    main()
