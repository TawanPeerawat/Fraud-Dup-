import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai

# กำหนดค่า Streamlit
st.set_page_config(
    page_title="Image Similarity Checker with Gemini AI",
    page_icon="🖼️",
    layout="wide"
)

# อ่าน API Key จาก Streamlit secrets
@st.cache_data
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return None

# ฟังก์ชันคำนวณความเหมือนแบบง่าย
def calculate_basic_similarity(img1, img2):
    """คำนวณความเหมือนพื้นฐานด้วย numpy"""
    # แปลงเป็น array และปรับขนาด
    arr1 = np.array(img1)
    arr2 = np.array(img2.resize(img1.size))
    
    # คำนวณ Mean Squared Error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # แปลงเป็น similarity score (0-1)
    max_possible_mse = 255 ** 2
    similarity = 1 - (mse / max_possible_mse)
    
    return max(0, similarity)

# ฟังก์ชันคำนวณ Color Similarity
def calculate_color_similarity(img1, img2):
    """คำนวณความเหมือนของสีเฉลี่ย"""
    # คำนวณสีเฉลี่ยของแต่ละภาพ
    avg_color1 = np.mean(np.array(img1), axis=(0, 1))
    avg_color2 = np.mean(np.array(img2), axis=(0, 1))
    
    # คำนวณ Euclidean distance
    color_distance = np.sqrt(np.sum((avg_color1 - avg_color2) ** 2))
    
    # แปลงเป็น similarity (0-1)
    max_distance = np.sqrt(3 * (255 ** 2))  # ระยะทางสูงสุด
    similarity = 1 - (color_distance / max_distance)
    
    return max(0, similarity)

# ฟังก์ชันสำหรับเรียก Gemini API
def analyze_images_with_gemini(image1, image2, basic_score, color_score, api_key):
    """ใช้ Gemini AI วิเคราะห์ภาพและให้คำอธิบาย"""
    try:
        # กำหนดค่า Gemini API
        genai.configure(api_key=api_key)
        
        # สร้าง model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # สร้าง prompt
        prompt = f"""
        กรุณาวิเคราะห์ความเหมือนของภาพทั้งสองนี้อย่างละเอียดและเป็นระบบ:

        ข้อมูลการวิเคราะห์เบื้องต้น:
        - Basic Similarity Score: {basic_score:.4f} ({basic_score*100:.2f}%)
        - Color Similarity Score: {color_score:.4f} ({color_score*100:.2f}%)

        กรุณาวิเคราะห์และตอบคำถามต่อไปนี้:

        1. **การประเมินโดยรวม**: ภาพทั้งสองเหมือนกันหรือไม่? ให้คะแนนความเหมือนเป็นเปอร์เซ็นต์ตามการประเมินของคุณ

        2. **การวิเคราะห์รายละเอียด**:
           - วัตถุหรือเนื้อหาหลักในภาพคืออะไร?
           - มีสิ่งที่เหมือนกันในภาพทั้งสองหรือไม่?
           - มีสิ่งที่แตกต่างกันอย่างชัดเจนหรือไม่?

        3. **การวิเคราะห์ด้านเทคนิค**:
           - สี และโทนของภาพเป็นอย่างไร?
           - องค์ประกอบและการจัดวางในภาพ?
           - คุณภาพและความชัดของภาพ?

        4. **เหตุผลของคะแนน**: อธิบายว่าทำไมการวิเคราะห์พื้นฐานถึงได้คะแนนแบบนี้?

        5. **สรุปและข้อแนะนำ**: สรุปผลการเปรียบเทียบและแนะนำการใช้งาน

        กรุณาตอบเป็นภาษาไทย ใช้รูปแบบ Markdown และ Emoji เพื่อให้อ่านง่าย
        และให้คำตอบที่ละเอียดแต่เข้าใจง่าย
        """
        
        # ส่งภาพและ prompt ไปยัง Gemini
        response = model.generate_content([prompt, image1, image2])
        
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ เกิดข้อผิดพลาดในการเรียก Gemini API: {error_msg}")
        
        # Fallback response
        overall_score = (basic_score + color_score) / 2
        return f"""
        ## ⚠️ ข้อผิดพลาดในการเชื่อมต่อ Gemini AI

        **ข้อผิดพลาด:** {error_msg}

        ### 📊 ผลการวิเคราะห์พื้นฐาน
        
        **ความเหมือนโดยรวม:** {overall_score*100:.1f}%
        
        - **Basic Similarity:** {basic_score*100:.1f}%
        - **Color Similarity:** {color_score*100:.1f}%
        
        ### 💡 คำแนะนำ
        1. ตรวจสอบ API Key ใน Secrets
        2. ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
        3. ลองใหม่อีกครั้ง
        
        กรุณาแก้ไขปัญหาเพื่อใช้งาน AI Analysis ได้เต็มรูปแบบ
        """

# Main App
def main():
    st.title("🖼️ Image Similarity Checker with Gemini AI")
    st.markdown("### ตรวจสอบความเหมือนของรูปภาพด้วย Google Gemini AI")
    
    # ดึง API Key
    api_key = get_api_key()
    
    # แสดงสถานะ API Key
    if not api_key:
        st.error("❌ ไม่พบ Gemini API Key")
        
        with st.expander("🔧 วิธีการตั้งค่า API Key"):
            st.markdown("""
            ### ขั้นตอนการตั้งค่า GEMINI_API_KEY:
            
            **1. ได้ API Key:**
            - ไปที่ [Google AI Studio](https://aistudio.google.com/)
            - สร้างโปรเจคใหม่หรือเลือกโปรเจคที่มีอยู่
            - คลิก "Get API Key" 
            - สร้าง API Key ใหม่และ Copy
            
            **2. ตั้งค่าใน Streamlit Cloud:**
            - ไปที่การตั้งค่าแอป (App Settings)
            - เลือกแท็บ "Secrets"
            - เพิ่มข้อมูลนี้ในช่อง Secrets:
            
            ```toml
            GEMINI_API_KEY = "ใส่_API_KEY_ที่_Copy_มา"
            ```
            
            **3. บันทึกและรีสตาร์ท:**
            - กด Save
            - รอให้แอปรีสตาร์ทอัตโนมัติ
            
            ⚠️ **หมายเหตุ:** API Key ต้องมีรูปแบบ AIzaSy...
            """)
        
        st.stop()  # หยุดการทำงานถ้าไม่มี API Key
    else:
        st.success("✅ Gemini API Key พร้อมใช้งาน")
    
    # คำอธิบายแอป
    with st.expander("ℹ️ เกี่ยวกับแอปนี้"):
        st.markdown("""
        **Image Similarity Checker with Gemini AI** ใช้เทคโนโลยี AI ล่าสุดจาก Google ในการวิเคราะห์ความเหมือนของภาพ
        
        **ฟีเจอร์หลัก:**
        - 🔍 **การวิเคราะห์พื้นฐาน**: ใช้อัลกอริทึมการประมวลผลภาพ
        - 🤖 **AI Analysis**: ใช้ Gemini AI วิเคราะห์และอธิบายผลลัพธ์
        - 📊 **ผลลัพธ์ที่ละเอียด**: คะแนนเปอร์เซ็นต์และคำอธิบายเชิงลึก
        
        **วิธีการใช้งาน:**
        1. อัปโหลดภาพ 2 ภาพที่ต้องการเปรียบเทียบ
        2. กดปุ่ม "วิเคราะห์ความเหมือน"
        3. รับผลลัพธ์และคำอธิบายจาก AI
        """)
    
    # Sidebar
    st.sidebar.header("⚙️ ตัวเลือก")
    
    # แสดงสถานะการเชื่อมต่อ
    with st.sidebar:
        st.subheader("🔗 สถานะการเชื่อมต่อ")
        if api_key:
            st.success("✅ Gemini AI พร้อมใช้งาน")
        else:
            st.error("❌ ไม่สามารถเชื่อมต่อ Gemini AI")
        
        # ข้อมูล API
        with st.expander("📊 ข้อมูล API"):
            st.markdown("""
            **Model:** Gemini 1.5 Flash
            **Provider:** Google AI
            **Features:** Vision + Text Analysis
            """)
    
    # File uploaders
    st.subheader("📤 อัปโหลดรูปภาพ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 ภาพที่ 1**")
        uploaded_file1 = st.file_uploader(
            "เลือกภาพแรก",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            key="img1",
            help="รองรับไฟล์: PNG, JPG, JPEG, GIF, BMP, WebP"
        )
        
    with col2:
        st.markdown("**📷 ภาพที่ 2**")
        uploaded_file2 = st.file_uploader(
            "เลือกภาพที่สอง",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
            key="img2",
            help="รองรับไฟล์: PNG, JPG, JPEG, GIF, BMP, WebP"
        )
    
    # แสดงภาพที่อัปโหลด
    if uploaded_file1 is not None and uploaded_file2 is not None:
        
        # โหลดภาพ
        try:
            image1 = Image.open(uploaded_file1).convert('RGB')
            image2 = Image.open(uploaded_file2).convert('RGB')
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดภาพได้: {str(e)}")
            st.stop()
        
        # แสดงภาพ
        st.subheader("🖼️ ภาพที่เลือก")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image1, caption=f"ภาพที่ 1: {uploaded_file1.name}", use_column_width=True)
            st.caption(f"ขนาด: {image1.size[0]} × {image1.size[1]} px")
            
        with col2:
            st.image(image2, caption=f"ภาพที่ 2: {uploaded_file2.name}", use_column_width=True)
            st.caption(f"ขนาด: {image2.size[0]} × {image2.size[1]} px")
        
        # ปุ่มวิเคราะห์
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🔍 วิเคราะห์ความเหมือนด้วย Gemini AI", 
                type="primary", 
                use_container_width=True
            ):
                # เริ่มการวิเคราะห์
                with st.spinner("🤖 กำลังใช้ Gemini AI วิเคราะห์ภาพ..."):
                    
                    # คำนวณคะแนนพื้นฐาน
                    basic_score = calculate_basic_similarity(image1, image2)
                    color_score = calculate_color_similarity(image1, image2)
                    
                    # แสดงผลคะแนนพื้นฐาน
                    st.subheader("⚡ ผลการวิเคราะห์เบื้องต้น")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "🔧 Basic Similarity",
                            f"{basic_score:.3f}",
                            f"{basic_score*100:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "🎨 Color Similarity", 
                            f"{color_score:.3f}",
                            f"{color_score*100:.1f}%"
                        )
                    
                    with col3:
                        overall_basic = (basic_score + color_score) / 2
                        st.metric(
                            "📊 Average",
                            f"{overall_basic:.3f}",
                            f"{overall_basic*100:.1f}%"
                        )
                    
                    # Progress bars
                    st.progress(basic_score, text=f"Basic: {basic_score*100:.1f}%")
                    st.progress(color_score, text=f"Color: {color_score*100:.1f}%")
                    
                # การวิเคราะห์ด้วย Gemini AI
                with st.spinner("🧠 กำลังให้ Gemini AI วิเคราะห์เชิงลึก..."):
                    ai_analysis = analyze_images_with_gemini(
                        image1, image2, basic_score, color_score, api_key
                    )
                
                # แสดงผลการวิเคราะห์ AI
                st.subheader("🤖 การวิเคราะห์โดย Gemini AI")
                st.markdown(ai_analysis)
                
                # ข้อมูลเพิ่มเติม
                with st.expander("📋 ข้อมูลเพิ่มเติม"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📷 ข้อมูลภาพที่ 1:**")
                        st.write(f"- ชื่อไฟล์: {uploaded_file1.name}")
                        st.write(f"- ขนาด: {image1.size[0]} × {image1.size[1]} pixels")
                        st.write(f"- โหมดสี: {image1.mode}")
                        st.write(f"- ขนาดไฟล์: {uploaded_file1.size:,} bytes")
                        
                    with col2:
                        st.markdown("**📷 ข้อมูลภาพที่ 2:**")
                        st.write(f"- ชื่อไฟล์: {uploaded_file2.name}")
                        st.write(f"- ขนาด: {image2.size[0]} × {image2.size[1]} pixels")
                        st.write(f"- โหมดสี: {image2.mode}")
                        st.write(f"- ขนาดไฟล์: {uploaded_file2.size:,} bytes")
                    
                    st.markdown("**🔬 เทคนิคการวิเคราะห์:**")
                    st.write("- **Basic Similarity:** Mean Squared Error + Similarity Conversion")
                    st.write("- **Color Similarity:** Average Color Comparison + Euclidean Distance")
                    st.write("- **AI Analysis:** Google Gemini 1.5 Flash Vision Model")
                
    else:
        # คำแนะนำเมื่อยังไม่ได้อัปโหลดภาพ
        st.info("👆 กรุณาอัปโหลดภาพ 2 ภาพเพื่อเริ่มการเปรียบเทียบ")
        
        # ตัวอย่างการใช้งาน
        with st.expander("💡 ตัวอย่างการใช้งาน"):
            st.markdown("""
            ### แอปนี้เหมาะสำหรับ:
            
            1. **ตรวจสอบภาพซ้ำ** - หาภาพที่เหมือนกันในคอลเลกชัน
            2. **เปรียบเทียบการแก้ไขภาพ** - ดูว่าภาพถูกแก้ไขมากน้อยแค่ไหน
            3. **ตรวจสอบคุณภาพภาพ** - เปรียบเทียบภาพต้นฉบับกับภาพที่บีบอัด
            4. **การวิจัยและการศึกษา** - วิเคราะห์ความเหมือนของภาพเพื่อการวิจัย
            
            ### เคล็ดลับการใช้งาน:
            - ใช้ภาพที่มีขนาดไม่เกิน 10MB เพื่อความเร็วในการประมวลผล
            - ภาพที่มีความละเอียดสูงจะให้ผลลัพธ์ที่แม่นยำมากขึ้น
            - AI จะวิเคราะห์ทั้งเนื้อหา สี รูปร่าง และบริบทของภาพ
            """)

if __name__ == "__main__":
    main()
