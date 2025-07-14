import streamlit as st
import numpy as np
from PIL import Image
import base64
import io
import google.generativeai as genai

# อ่าน API Key จาก Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    GEMINI_API_KEY = None

# กำหนดค่า Streamlit
st.set_page_config(
    page_title="Image Similarity Checker",
    page_icon="🖼️",
    layout="wide"
)

# ฟังก์ชันคำนวณความเหมือนแบบง่าย (Histogram Comparison)
def calculate_histogram_similarity(img1, img2):
    """คำนวณความเหมือนจาก histogram ของสี"""
    # แปลงเป็น array
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    # ปรับขนาดให้เหมือนกัน
    if arr1.shape != arr2.shape:
        img2_resized = img2.resize(img1.size)
        arr2 = np.array(img2_resized)
    
    # คำนวณ histogram สำหรับแต่ละ channel
    hist1_r = np.histogram(arr1[:,:,0], bins=50, range=(0, 255))[0]
    hist1_g = np.histogram(arr1[:,:,1], bins=50, range=(0, 255))[0]
    hist1_b = np.histogram(arr1[:,:,2], bins=50, range=(0, 255))[0]
    
    hist2_r = np.histogram(arr2[:,:,0], bins=50, range=(0, 255))[0]
    hist2_g = np.histogram(arr2[:,:,1], bins=50, range=(0, 255))[0]
    hist2_b = np.histogram(arr2[:,:,2], bins=50, range=(0, 255))[0]
    
    # คำนวณ correlation coefficient
    corr_r = np.corrcoef(hist1_r, hist2_r)[0,1]
    corr_g = np.corrcoef(hist1_g, hist2_g)[0,1]
    corr_b = np.corrcoef(hist1_b, hist2_b)[0,1]
    
    # เฉลี่ยของ 3 channels
    similarity = (corr_r + corr_g + corr_b) / 3
    
    # จัดการ NaN values
    if np.isnan(similarity):
        similarity = 0.0
    
    return max(0, similarity)  # ให้อยู่ในช่วง 0-1

# ฟังก์ชันคำนวณความเหมือนจาก pixel values
def calculate_pixel_similarity(img1, img2):
    """คำนวณความเหมือนจาก pixel values โดยตรง"""
    # แปลงเป็น array และปรับขนาด
    arr1 = np.array(img1)
    arr2 = np.array(img2.resize(img1.size))
    
    # คำนวณ Mean Squared Error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # แปลงเป็น similarity score (0-1)
    max_possible_mse = 255 ** 2
    similarity = 1 - (mse / max_possible_mse)
    
    return max(0, similarity)

# ฟังก์ชันสำหรับเรียก Gemini API
def get_ai_analysis(image1, image2, hist_score, pixel_score):
    try:
        # ตรวจสอบ API Key
        if not GEMINI_API_KEY:
            st.error("⚠️ กรุณาตั้งค่า Gemini API Key ในการตั้งค่า Streamlit Cloud")
            return get_fallback_response(hist_score, pixel_score)
        
        # กำหนดค่า Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        
        # สร้าง model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # สร้าง prompt
        prompt = f"""
        วิเคราะห์ความเหมือนของภาพทั้งสองนี้อย่างละเอียด:
        
        ผลการวิเคราะห์เบื้องต้น:
        - Histogram Similarity: {hist_score:.4f} ({hist_score*100:.2f}%)
        - Pixel Similarity: {pixel_score:.4f} ({pixel_score*100:.2f}%)
        
        กรุณาวิเคราะห์และอธิบายผลลัพธ์อย่างละเอียดว่า:
        1. ภาพทั้งสองเหมือนกันหรือไม่ และเหมือนกันกี่เปอร์เซ็นต์ตามการประเมินของคุณ
        2. จุดที่เหมือนกันและแตกต่างกันคืออะไร (สี, รูปร่าง, เนื้อหา, การจัดวาง)
        3. เหตุผลที่ทำให้ได้คะแนนการวิเคราะห์แบบนี้
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
        return get_fallback_response(hist_score, pixel_score)

# ฟังก์ชัน fallback response
def get_fallback_response(hist_score, pixel_score):
    overall_score = (hist_score + pixel_score) / 2
    
    fallback_response = f"""
    ## 📊 การวิเคราะห์ความเหมือนของภาพ
    
    **ผลลัพธ์โดยรวม:** ภาพทั้งสองมีความเหมือนกัน **{overall_score*100:.1f}%**
    
    ### 🎨 การวิเคราะห์สี (Histogram: {hist_score*100:.1f}%)
    - วิเคราะห์การกระจายของสีในภาพ
    - {"คะแนนสูง แสดงว่าภาพมีโทนสีคล้ายกัน" if hist_score > 0.7 else "คะแนนปานกลาง สีในภาพค่อนข้างแตกต่าง" if hist_score > 0.3 else "คะแนนต่ำ ภาพมีสีที่แตกต่างกันมาก"}
    
    ### 🖼️ การวิเคราะห์ Pixel (Pixel Similarity: {pixel_score*100:.1f}%)
    - วิเคราะห์ความแตกต่างของ pixel แต่ละจุด
    - {"คะแนนสูง แสดงว่าภาพมีรายละเอียดคล้ายกัน" if pixel_score > 0.7 else "คะแนนปานกลาง รายละเอียดค่อนข้างแตกต่าง" if pixel_score > 0.3 else "คะแนนต่ำ ภาพมีรายละเอียดที่แตกต่างกันมาก"}
    
    ### 📈 สรุปผลการเปรียบเทียบ
    - **ความเหมือนโดยรวม:** {overall_score*100:.1f}%
    - **ความเหมือนด้านสี:** {hist_score*100:.1f}%
    - **ความเหมือนด้านรายละเอียด:** {pixel_score*100:.1f}%
    
    ### 💡 คำแนะนำ
    {"ภาพทั้งสองมีความเหมือนกันสูงมาก น่าจะเป็นภาพเดียวกันหรือภาพที่มีเนื้อหาเหมือนกันมาก" if overall_score > 0.8 else "ภาพมีความเหมือนปานกลาง อาจเป็นภาพเดียวกันที่ผ่านการแก้ไขหรือภาพที่คล้ายกัน" if overall_score > 0.5 else "ภาพทั้งสองแตกต่างกันมาก น่าจะเป็นภาพต่างกัน"}
    
    ---
    *หมายเหตุ: การวิเคราะห์นี้ใช้วิธีการพื้นฐาน เพื่อความแม่นยำมากขึ้นควรใช้ AI analysis*
    """
    
    return fallback_response

# Main App
def main():
    st.title("🖼️ Image Similarity Checker with AI Analysis")
    st.markdown("### ตรวจสอบความเหมือนของรูปภาพด้วย AI (เวอร์ชันง่าย)")
    
    # แสดงสถานะ API Key
    if not GEMINI_API_KEY:
        st.warning("⚠️ กรุณาตั้งค่า Gemini API Key ในการตั้งค่า Streamlit Cloud")
        with st.expander("📝 วิธีการตั้งค่า API Key"):
            st.markdown("""
            1. ไปที่การตั้งค่าแอป (Settings)
            2. เลือก "Secrets"
            3. เพิ่มข้อมูลนี้:
            ```
            GEMINI_API_KEY = "ใส่_API_KEY_ของคุณ_ที่นี่"
            ```
            4. บันทึกและรีสตาร์ทแอป
            """)
    else:
        st.success("✅ Gemini API Key ตั้งค่าแล้ว")
    
    # Sidebar
    st.sidebar.header("ตัวเลือก")
    
    analysis_method = st.sidebar.selectbox(
        "เลือกวิธีการวิเคราะห์",
        ["ทั้งหมด", "Histogram เท่านั้น", "Pixel เท่านั้น"]
    )
    
    # วิธีการได้ API Key
    with st.sidebar.expander("📝 วิธีการได้ API Key"):
        st.markdown("""
        1. ไปที่ [Google AI Studio](https://aistudio.google.com/)
        2. คลิก "Get API Key"
        3. สร้าง API Key ใหม่
        4. นำไปใส่ในการตั้งค่า Streamlit Cloud
        """)
    
    # คำอธิบายวิธีการวิเคราะห์
    with st.sidebar.expander("ℹ️ วิธีการวิเคราะห์"):
        st.markdown("""
        **Histogram Similarity:**
        - วิเคราะห์การกระจายของสีในภาพ
        - เหมาะสำหรับเปรียบเทียบโทนสี
        
        **Pixel Similarity:**
        - เปรียบเทียบค่า pixel ตรงๆ
        - เหมาะสำหรับภาพที่คล้ายกันมาก
        """)
    
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
                
                hist_score = None
                pixel_score = None
                
                # คำนวณ Histogram similarity
                if analysis_method in ["ทั้งหมด", "Histogram เท่านั้น"]:
                    hist_score = calculate_histogram_similarity(image1, image2)
                
                # คำนวณ Pixel similarity
                if analysis_method in ["ทั้งหมด", "Pixel เท่านั้น"]:
                    pixel_score = calculate_pixel_similarity(image1, image2)
                
                # แสดงผลลัพธ์
                st.subheader("📊 ผลการวิเคราะห์")
                
                # แสดงคะแนน
                col1, col2, col3 = st.columns(3)
                
                if hist_score is not None:
                    with col1:
                        st.metric(
                            "Histogram Similarity",
                            f"{hist_score:.4f}",
                            f"{hist_score*100:.2f}%"
                        )
                
                if pixel_score is not None:
                    with col2:
                        st.metric(
                            "Pixel Similarity",
                            f"{pixel_score:.4f}",
                            f"{pixel_score*100:.2f}%"
                        )
                
                if hist_score is not None and pixel_score is not None:
                    overall_score = (hist_score + pixel_score) / 2
                    with col3:
                        st.metric(
                            "Overall Similarity",
                            f"{overall_score:.4f}",
                            f"{overall_score*100:.2f}%"
                        )
                
                # Progress bars
                if hist_score is not None:
                    st.subheader("📈 Visual Progress")
                    st.progress(hist_score, text=f"Histogram: {hist_score*100:.1f}%")
                
                if pixel_score is not None:
                    st.progress(pixel_score, text=f"Pixel: {pixel_score*100:.1f}%")
                
                # AI Analysis
                st.subheader("🤖 AI Analysis")
                
                with st.spinner("กำลังให้ AI วิเคราะห์ภาพ..."):
                    # เรียก Gemini API analysis
                    ai_analysis = get_ai_analysis(
                        image1, 
                        image2, 
                        hist_score or 0, 
                        pixel_score or 0
                    )
                    
                    st.markdown(ai_analysis)
                
                # เพิ่มข้อมูลทางเทคนิค
                with st.expander("ข้อมูลทางเทคนิค"):
                    st.write("**ข้อมูลภาพที่ 1:**")
                    st.write(f"- ขนาด: {image1.size}")
                    st.write(f"- โหมดสี: {image1.mode}")
                    
                    st.write("**ข้อมูลภาพที่ 2:**")
                    st.write(f"- ขนาด: {image2.size}")
                    st.write(f"- โหมดสี: {image2.mode}")
                    
                    st.write("**วิธีการวิเคราะห์:**")
                    st.write("- Histogram Similarity: เปรียบเทียบการกระจายของสี")
                    st.write("- Pixel Similarity: เปรียบเทียบค่า pixel โดยตรง")
                    st.write("- AI Analysis: Google Gemini 1.5 Flash")

if __name__ == "__main__":
    main()
