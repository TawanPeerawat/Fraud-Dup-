import streamlit as st
import numpy as np
from PIL import Image
import requests
import json
import base64
import io

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Image Similarity Checker with Gemini AI", 
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Similarity Checker")
st.markdown("### ตรวจสอบความเหมือนของรูปภาพด้วย Gemini AI")

# ==============================
# Gemini API Configuration
# ==============================
# วิธีที่ 1: ใช้ Secrets (แนะนำ)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.success("✅ Gemini API Key loaded from secrets successfully!")
except:
    # วิธีที่ 2: ใส่ API Key ตรงนี้ (สำรอง)
    GEMINI_API_KEY = "AIzaSyDhcBaFpk3YqRJtb6kLfQhbJSnGoklha8o"  # ← ใส่ API Key ของคุณตรงนี้
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        st.error("❌ กรุณาใส่ Gemini API Key ในไฟล์หรือใน Secrets")
        st.stop()
    else:
        st.success("✅ Gemini API Key loaded from code successfully!")

# ==============================
# Helper Functions
# ==============================
def image_to_base64(image):
    """แปลงรูปภาพเป็น base64 สำหรับส่งไปยัง API"""
    buffered = io.BytesIO()
    # ปรับขนาดภาพถ้าใหญ่เกินไป (เพื่อประสิทธิภาพ)
    if image.size[0] > 1024 or image.size[1] > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

def calculate_basic_similarity(img1, img2):
    """คำนวณความเหมือนพื้นฐานด้วย Mean Squared Error"""
    # ปรับขนาดให้เท่ากัน
    size = (256, 256)  # ใช้ขนาดมาตรฐานเพื่อเปรียบเทียบ
    img1_resized = img1.resize(size)
    img2_resized = img2.resize(size)
    
    # แปลงเป็น array
    arr1 = np.array(img1_resized, dtype=np.float32)
    arr2 = np.array(img2_resized, dtype=np.float32)
    
    # คำนวณ MSE
    mse = np.mean((arr1 - arr2) ** 2)
    
    # แปลงเป็น similarity (0-1)
    max_mse = 255.0 ** 2
    similarity = 1 - (mse / max_mse)
    
    return max(0, min(1, similarity))

def calculate_color_similarity(img1, img2):
    """คำนวณความเหมือนของสีเฉลี่ย"""
    # คำนวณสีเฉลี่ย
    avg_color1 = np.mean(np.array(img1.resize((64, 64))), axis=(0, 1))
    avg_color2 = np.mean(np.array(img2.resize((64, 64))), axis=(0, 1))
    
    # คำนวณระยะทาง Euclidean
    distance = np.sqrt(np.sum((avg_color1 - avg_color2) ** 2))
    
    # แปลงเป็น similarity
    max_distance = np.sqrt(3 * (255 ** 2))
    similarity = 1 - (distance / max_distance)
    
    return max(0, min(1, similarity))

def calculate_histogram_similarity(img1, img2):
    """คำนวณความเหมือนจาก histogram"""
    # ปรับขนาดเพื่อประสิทธิภาพ
    img1_small = img1.resize((128, 128))
    img2_small = img2.resize((128, 128))
    
    arr1 = np.array(img1_small)
    arr2 = np.array(img2_small)
    
    # คำนวณ histogram สำหรับแต่ละ channel
    hist1_r = np.histogram(arr1[:,:,0], bins=32, range=(0, 255))[0]
    hist1_g = np.histogram(arr1[:,:,1], bins=32, range=(0, 255))[0]
    hist1_b = np.histogram(arr1[:,:,2], bins=32, range=(0, 255))[0]
    
    hist2_r = np.histogram(arr2[:,:,0], bins=32, range=(0, 255))[0]
    hist2_g = np.histogram(arr2[:,:,1], bins=32, range=(0, 255))[0]
    hist2_b = np.histogram(arr2[:,:,2], bins=32, range=(0, 255))[0]
    
    # คำนวณ correlation
    try:
        corr_r = np.corrcoef(hist1_r, hist2_r)[0,1] if np.std(hist1_r) > 0 and np.std(hist2_r) > 0 else 0
        corr_g = np.corrcoef(hist1_g, hist2_g)[0,1] if np.std(hist1_g) > 0 and np.std(hist2_g) > 0 else 0
        corr_b = np.corrcoef(hist1_b, hist2_b)[0,1] if np.std(hist1_b) > 0 and np.std(hist2_b) > 0 else 0
        
        # เฉลี่ยและจัดการ NaN
        correlations = [corr_r, corr_g, corr_b]
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        
        if valid_correlations:
            similarity = np.mean(valid_correlations)
        else:
            similarity = 0.0
            
        return max(0, min(1, similarity))
    except:
        return 0.0

def call_gemini_api(prompt_text, image1, image2):
    """เรียก Gemini API เพื่อวิเคราะห์ภาพ"""
    try:
        # แปลงภาพเป็น base64
        img1_base64 = image_to_base64(image1)
        img2_base64 = image_to_base64(image2)
        
        # API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # สร้าง payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img1_base64
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg", 
                                "data": img2_base64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048
            }
        }
        
        # เรียก API
        with st.spinner("🤖 กำลังให้ Gemini AI วิเคราะห์ภาพ..."):
            response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    return content["parts"][0]["text"]
                else:
                    return "❌ ไม่ได้รับการตอบสนองที่สมบูรณ์จาก Gemini AI"
            else:
                return "❌ ไม่ได้รับการตอบสนองจาก Gemini AI"
        elif response.status_code == 429:
            return "❌ API Rate Limit: กรุณารอสักครู่แล้วลองใหม่"
        elif response.status_code == 400:
            return "❌ API Error: ข้อมูลที่ส่งไปไม่ถูกต้อง"
        elif response.status_code == 403:
            return "❌ API Error: API Key ไม่ถูกต้องหรือไม่มีสิทธิ์เข้าถึง"
        else:
            return f"❌ API Error: {response.status_code} - {response.text[:200]}..."
            
    except requests.exceptions.Timeout:
        return "❌ เกิดข้อผิดพลาด: การเชื่อมต่อหมดเวลา กรุณาลองใหม่"
    except requests.exceptions.ConnectionError:
        return "❌ เกิดข้อผิดพลาด: ไม่สามารถเชื่อมต่อกับ API ได้"
    except Exception as e:
        return f"❌ เกิดข้อผิดพลาด: {str(e)}"

# ==============================
# Session State
# ==============================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ==============================
# Main UI
# ==============================

# คำอธิบายแอป
with st.expander("ℹ️ เกี่ยวกับแอปนี้"):
    st.markdown("""
    **Image Similarity Checker** ใช้เทคโนโลยี AI จาก Google Gemini ในการวิเคราะห์ความเหมือนของภาพ
    
    **ฟีเจอร์หลัก:**
    - 🔍 **การวิเคราะห์เบื้องต้น**: Basic, Color, และ Histogram Similarity
    - 🤖 **AI Analysis**: ใช้ Gemini AI วิเคราะห์และอธิบายผลลัพธ์เชิงลึก
    - 📊 **ผลลัพธ์ครบถ้วน**: คะแนนเปอร์เซ็นต์พร้อมคำอธิบายที่เข้าใจง่าย
    
    **วิธีการใช้งาน:**
    1. อัปโหลดภาพ 2 ภาพที่ต้องการเปรียบเทียบ
    2. กดปุ่ม "วิเคราะห์ความเหมือน"
    3. รับผลลัพธ์และคำอธิบายจาก Gemini AI
    """)

# อัปโหลดไฟล์
st.subheader("📤 อัปโหลดรูปภาพเพื่อเปรียบเทียบ")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📷 ภาพที่ 1**")
    uploaded_file1 = st.file_uploader(
        "เลือกภาพแรก",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        key="img1",
        help="รองรับไฟล์: PNG, JPG, JPEG, GIF, BMP, WebP (ขนาดสูงสุด 200MB)"
    )
    
with col2:
    st.markdown("**📷 ภาพที่ 2**")
    uploaded_file2 = st.file_uploader(
        "เลือกภาพที่สอง",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        key="img2",
        help="รองรับไฟล์: PNG, JPG, JPEG, GIF, BMP, WebP (ขนาดสูงสุด 200MB)"
    )

# ==============================
# Image Processing
# ==============================
if uploaded_file1 is not None and uploaded_file2 is not None:
    
    # โหลดและตรวจสอบภาพ
    try:
        image1 = Image.open(uploaded_file1).convert('RGB')
        image2 = Image.open(uploaded_file2).convert('RGB')
    except Exception as e:
        st.error(f"❌ ไม่สามารถโหลดภาพได้: {str(e)}")
        st.stop()
    
    # แสดงภาพที่อัปโหลด
    st.subheader("🖼️ ภาพที่เลือกสำหรับเปรียบเทียบ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption=f"📷 ภาพที่ 1: {uploaded_file1.name}", use_column_width=True)
        st.caption(f"ขนาด: {image1.size[0]} × {image1.size[1]} pixels | ไฟล์: {uploaded_file1.size:,} bytes")
        
    with col2:
        st.image(image2, caption=f"📷 ภาพที่ 2: {uploaded_file2.name}", use_column_width=True)
        st.caption(f"ขนาด: {image2.size[0]} × {image2.size[1]} pixels | ไฟล์: {uploaded_file2.size:,} bytes")
    
    # ปุ่มวิเคราะห์
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 วิเคราะห์ความเหมือนด้วย Gemini AI", 
            type="primary", 
            use_container_width=True
        )
    
    if analyze_button:
        # ==============================
        # Analysis Process
        # ==============================
        
        # ขั้นตอนที่ 1: การวิเคราะห์เบื้องต้น
        with st.spinner("⚡ กำลังคำนวณความเหมือนเบื้องต้น..."):
            basic_score = calculate_basic_similarity(image1, image2)
            color_score = calculate_color_similarity(image1, image2)
            histogram_score = calculate_histogram_similarity(image1, image2)
            overall_score = (basic_score + color_score + histogram_score) / 3
        
        # แสดงผลเบื้องต้น
        st.subheader("⚡ ผลการวิเคราะห์เบื้องต้น")
        
        col1, col2, col3, col4 = st.columns(4)
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
            st.metric(
                "📊 Histogram Similarity", 
                f"{histogram_score:.3f}",
                f"{histogram_score*100:.1f}%"
            )
        with col4:
            st.metric(
                "⭐ Overall Score",
                f"{overall_score:.3f}",
                f"{overall_score*100:.1f}%"
            )
        # Progress bars แสดงความเหมือน
st.markdown("### 📊 แสดงผลแบบกราฟิก")

# Basic Similarity
st.markdown(f"**🔧 Basic Similarity: {basic_score*100:.1f}%**")
st.progress(int(basic_score * 100))

# Color Similarity  
st.markdown(f"**🎨 Color Similarity: {color_score*100:.1f}%**")
st.progress(int(color_score * 100))

# Histogram Similarity
st.markdown(f"**📊 Histogram Similarity: {histogram_score*100:.1f}%**")
st.progress(int(histogram_score * 100))

# Overall Score
st.markdown(f"**⭐ Overall Score: {overall_score*100:.1f}%**")
st.progress(int(overall_score * 100))
        
        # ขั้นตอนที่ 2: การวิเคราะห์ด้วย Gemini AI
        st.subheader("🤖 การวิเคราะห์เชิงลึกโดย Gemini AI")
        
        # สร้าง prompt สำหรับ Gemini
        prompt = f"""
        กรุณาวิเคราะห์ความเหมือนของภาพทั้งสองนี้อย่างละเอียดและเป็นระบบ:

        ข้อมูลการวิเคราะห์เบื้องต้น:
        - Basic Similarity: {basic_score:.4f} ({basic_score*100:.2f}%)
        - Color Similarity: {color_score:.4f} ({color_score*100:.2f}%)
        - Histogram Similarity: {histogram_score:.4f} ({histogram_score*100:.2f}%)
        - Overall Score: {overall_score:.4f} ({overall_score*100:.2f}%)

        กรุณาวิเคราะห์และตอบคำถามต่อไปนี้อย่างละเอียด:

        ## 1. การประเมินโดยรวม
        - ภาพทั้งสองเหมือนกันหรือไม่? 
        - ให้คะแนนความเหมือนเป็นเปอร์เซ็นต์ตามการประเมินของคุณ
        - ระดับความเชื่อมั่นในการประเมิน

        ## 2. การวิเคราะห์เนื้อหาภาพ
        - วัตถุ/บุคคล/สิ่งของหลักในภาพแต่ละภาพคืออะไร?
        - มีสิ่งที่เหมือนกันในภาพทั้งสองหรือไม่? ระบุรายละเอียด
        - มีสิ่งที่แตกต่างกันอย่างชัดเจนหรือไม่? ระบุรายละเอียด

        ## 3. การวิเคราะห์ด้านเทคนิค
        - สีและโทนของภาพเป็นอย่างไร? (สว่าง/มืด, อบอุ่น/เย็น)
        - องค์ประกอบและการจัดวางในภาพ (composition)
        - คุณภาพและความชัดของภาพ
        - มุมกล้องและการถ่ายภาพ

        ## 4. เหตุผลของคะแนน
        - อธิบายว่าทำไมการวิเคราะห์พื้นฐานถึงได้คะแนนแบบนี้?
        - คะแนนส่วนไหนสูง/ต่ำ และเพราะอะไร?
        - ความสอดคล้องระหว่างคะแนนต่างๆ

        ## 5. สรุปและข้อแนะนำ
        - สรุปผลการเปรียบเทียบในภาพรวม
        - ข้อแนะนำสำหรับการใช้งานต่อ
        - ข้อจำกัดของการวิเคราะห์นี้

        กรุณาตอบเป็นภาษาไทย ใช้รูปแบบ Markdown ที่อ่านง่าย มีการใช้ Emoji ที่เหมาะสม 
        และให้คำตอบที่ละเอียดแต่เข้าใจง่าย เน้นความเป็นประโยชน์ต่อผู้ใช้
        """
        
        # เรียก Gemini API
        ai_analysis = call_gemini_api(prompt, image1, image2)
        
        # แสดงผลการวิเคราะห์ AI
        st.markdown(ai_analysis)
        
        # บันทึกประวัติ
        analysis_record = {
            "image1_name": uploaded_file1.name,
            "image2_name": uploaded_file2.name,
            "basic_score": basic_score,
            "color_score": color_score,
            "histogram_score": histogram_score,
            "overall_score": overall_score,
            "ai_analysis_preview": ai_analysis[:200] + "..." if len(ai_analysis) > 200 else ai_analysis
        }
        st.session_state.analysis_history.append(analysis_record)
        
        # ข้อมูลเพิ่มเติม
        with st.expander("📋 ข้อมูลเพิ่มเติมและเทคนิค"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📷 รายละเอียดภาพที่ 1:**")
                st.write(f"- ชื่อไฟล์: {uploaded_file1.name}")
                st.write(f"- ขนาด: {image1.size[0]} × {image1.size[1]} pixels")
                st.write(f"- โหมดสี: {image1.mode}")
                st.write(f"- ขนาดไฟล์: {uploaded_file1.size:,} bytes")
                st.write(f"- อัตราส่วน: {image1.size[0]/image1.size[1]:.2f}:1")
                
            with col2:
                st.markdown("**📷 รายละเอียดภาพที่ 2:**")
                st.write(f"- ชื่อไฟล์: {uploaded_file2.name}")
                st.write(f"- ขนาด: {image2.size[0]} × {image2.size[1]} pixels")
                st.write(f"- โหมดสี: {image2.mode}")
                st.write(f"- ขนาดไฟล์: {uploaded_file2.size:,} bytes")
                st.write(f"- อัตราส่วน: {image2.size[0]/image2.size[1]:.2f}:1")
            
            st.markdown("**🔬 เทคนิคการวิเคราะห์:**")
            st.write("- **Basic Similarity:** คำนวณจาก Mean Squared Error ระหว่าง pixel values")
            st.write("- **Color Similarity:** เปรียบเทียบสีเฉลี่ยโดยรวมด้วย Euclidean Distance")
            st.write("- **Histogram Similarity:** วิเคราะห์การกระจายของสีด้วย Correlation Coefficient")
            st.write("- **AI Analysis:** Google Gemini 1.5 Flash Vision Model ผ่าน REST API")

else:
    # แสดงคำแนะนำเมื่อยังไม่ได้อัปโหลดภาพ
    st.info("👆 กรุณาอัปโหลดภาพ 2 ภาพเพื่อเริ่มการเปรียบเทียบความเหมือน")
    
    # ตัวอย่างการใช้งาน
    with st.expander("💡 ตัวอย่างการใช้งานและคำแนะนำ"):
        st.markdown("""
        ### 🎯 แอปนี้เหมาะสำหรับ:
        
        1. **🔍 ตรวจสอบภาพซ้ำ** - หาภาพที่เหมือนกันในคอลเลกชัน
        2. **✏️ เปรียบเทียบการแก้ไขภาพ** - ดูว่าภาพถูกแก้ไขมากน้อยแค่ไหน
        3. **📏 ตรวจสอบคุณภาพภาพ** - เปรียบเทียบภาพต้นฉบับกับภาพที่บีบอัด
        4. **🎓 การวิจัยและการศึกษา** - วิเคราะห์ความเหมือนของภาพเพื่อการวิจัย
        5. **👤 เปรียบเทียบภาพบุคคล** - ตรวจสอบว่าเป็นบุคคลเดียวกันหรือไม่
        
        ### 💡 เคล็ดลับการใช้งาน:
        
        - **📐 ขนาดไฟล์:** ใช้ภาพขนาดไม่เกิน 10MB เพื่อความเร็วในการประมวลผล
        - **🖼️ ความละเอียด:** ภาพที่มีความละเอียดสูงจะให้ผลลัพธ์ที่แม่นยำมากขึ้น
        - **🎨 รูปแบบไฟล์:** รองรับ JPG, PNG, GIF, BMP, WebP
        - **🤖 AI Analysis:** Gemini AI จะวิเคราะห์ทั้งเนื้อหา สี รูปร่าง และบริบทของภาพ
        - **⏱️ เวลาประมวลผล:** การวิเคราะห์ใช้เวลาประมาณ 10-30 วินาที ขึ้นอยู่กับขนาดภาพ
        
        ### 📊 การตีความผลลัพธ์:
        
        - **90-100%:** ภาพเหมือนกันมาก (อาจเป็นภาพเดียวกัน)
        - **70-89%:** ภาพคล้ายกันสูง (อาจเป็นภาพเดียวกันที่แก้ไข)
        - **50-69%:** ภาพคล้ายกันปานกลาง (มีลักษณะร่วมกัน)
        - **30-49%:** ภาพคล้ายกันน้อย (มีจุดร่วมบางประการ)
        - **0-29%:** ภาพแตกต่างกันมาก (เป็นภาพต่างกัน)
        """)

# ==============================
# Analysis History
# ==============================
if st.session_state.analysis_history:
    st.markdown("---")
    with st.expander(f"📈 ประวัติการวิเคราะห์ ({len(st.session_state.analysis_history)} ครั้ง)"):
        for i, record in enumerate(reversed(st.session_state.analysis_history[-5:])):  # แสดงแค่ 5 ครั้งล่าสุด
            idx = len(st.session_state.analysis_history) - i
            st.markdown(f"**🔍 การวิเคราะห์ครั้งที่ {idx}**")
            st.write(f"📷 {record['image1_name']} ⚡ {record['image2_name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Basic", f"{record['basic_score']:.3f}")
            with col2:
                st.metric("Color", f"{record['color_score']:.3f}")
            with col3:
                st.metric("Histogram", f"{record['histogram_score']:.3f}")
            with col4:
                st.metric("Overall", f"{record['overall_score']:.3f}")
            
            st.caption(f"AI Analysis: {record['ai_analysis_preview']}")
            st.markdown("---")
        
        if len(st.session_state.analysis_history) > 5:
            st.info(f"📋 แสดง 5 ครั้งล่าสุดจากทั้งหมด {len(st.session_state.analysis_history)} ครั้ง")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 <strong>Image Similarity Checker</strong> | Powered by <strong>Google Gemini AI</strong></p>
    <p>Made with ❤️ using Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
