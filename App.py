import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai

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
# Gemini API Key: ใส่ตรงนี้
# ==============================
try:
    genai.configure(api_key="AIzaSyDhcBaFpk3YqRJtb6kLfQhbJSnGoklha8o")
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    st.success("Gemini API Key successfully configured!")
except Exception as e:
    st.error(f"Error setting up the Gemini model: {e}")
    st.stop()

# ==============================
# Session State Initialization
# ==============================
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = {"image1": None, "image2": None}

# ==============================
# Helper Functions
# ==============================
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

def calculate_histogram_similarity(img1, img2):
    """คำนวณความเหมือนจาก histogram ของสี"""
    # แปลงเป็น array
    arr1 = np.array(img1)
    arr2 = np.array(img2.resize(img1.size))
    
    # คำนวณ histogram สำหรับแต่ละ channel
    hist1_r = np.histogram(arr1[:,:,0], bins=50, range=(0, 255))[0]
    hist1_g = np.histogram(arr1[:,:,1], bins=50, range=(0, 255))[0]
    hist1_b = np.histogram(arr1[:,:,2], bins=50, range=(0, 255))[0]
    
    hist2_r = np.histogram(arr2[:,:,0], bins=50, range=(0, 255))[0]
    hist2_g = np.histogram(arr2[:,:,1], bins=50, range=(0, 255))[0]
    hist2_b = np.histogram(arr2[:,:,2], bins=50, range=(0, 255))[0]
    
    # คำนวณ correlation coefficient
    corr_r = np.corrcoef(hist1_r, hist2_r)[0,1] if len(set(hist1_r)) > 1 and len(set(hist2_r)) > 1 else 0
    corr_g = np.corrcoef(hist1_g, hist2_g)[0,1] if len(set(hist1_g)) > 1 and len(set(hist2_g)) > 1 else 0
    corr_b = np.corrcoef(hist1_b, hist2_b)[0,1] if len(set(hist1_b)) > 1 and len(set(hist2_b)) > 1 else 0
    
    # เฉลี่ยของ 3 channels
    similarity = (corr_r + corr_g + corr_b) / 3
    
    # จัดการ NaN values
    if np.isnan(similarity):
        similarity = 0.0
    
    return max(0, similarity)

# ==============================
# UI Layout
# ==============================

# คำอธิบายแอป
with st.expander("ℹ️ เกี่ยวกับแอปนี้"):
    st.markdown("""
    **Image Similarity Checker** ใช้เทคโนโลยี AI จาก Google Gemini ในการวิเคราะห์ความเหมือนของภาพ
    
    **ฟีเจอร์หลัก:**
    - 🔍 **การวิเคราะห์พื้นฐาน**: Basic, Color, และ Histogram Similarity
    - 🤖 **AI Analysis**: ใช้ Gemini AI วิเคราะห์และอธิบายผลลัพธ์
    - 📊 **ผลลัพธ์ที่ละเอียด**: คะแนนเปอร์เซ็นต์และคำอธิบายเชิงลึก
    
    **วิธีการใช้งาน:**
    1. อัปโหลดภาพ 2 ภาพที่ต้องการเปรียบเทียบ
    2. กดปุ่ม "วิเคราะห์ความเหมือน"
    3. รับผลลัพธ์และคำอธิบายจาก AI
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

# ==============================
# Image Processing and Display
# ==============================
if uploaded_file1 is not None and uploaded_file2 is not None:
    
    # โหลดภาพ
    try:
        image1 = Image.open(uploaded_file1).convert('RGB')
        image2 = Image.open(uploaded_file2).convert('RGB')
        
        # เก็บใน session state
        st.session_state.uploaded_images["image1"] = image1
        st.session_state.uploaded_images["image2"] = image2
        
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
            # ==============================
            # Analysis Process
            # ==============================
            
            if model:
                try:
                    # เริ่มการวิเคราะห์
                    with st.spinner("🤖 กำลังใช้ Gemini AI วิเคราะห์ภาพ..."):
                        
                        # คำนวณคะแนนพื้นฐาน
                        basic_score = calculate_basic_similarity(image1, image2)
                        color_score = calculate_color_similarity(image1, image2)
                        histogram_score = calculate_histogram_similarity(image1, image2)
                        
                        # แสดงผลคะแนนพื้นฐาน
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
                            overall_basic = (basic_score + color_score + histogram_score) / 3
                            st.metric(
                                "⭐ Average",
                                f"{overall_basic:.3f}",
                                f"{overall_basic*100:.1f}%"
                            )
                        
                        # Progress bars
                        st.progress(basic_score, text=f"Basic: {basic_score*100:.1f}%")
                        st.progress(color_score, text=f"Color: {color_score*100:.1f}%")
                        st.progress(histogram_score, text=f"Histogram: {histogram_score*100:.1f}%")
                        
                        # สร้าง prompt สำหรับ Gemini
                        prompt = f"""
                        กรุณาวิเคราะห์ความเหมือนของภาพทั้งสองนี้อย่างละเอียดและเป็นระบบ:

                        ข้อมูลการวิเคราะห์เบื้องต้น:
                        - Basic Similarity Score: {basic_score:.4f} ({basic_score*100:.2f}%)
                        - Color Similarity Score: {color_score:.4f} ({color_score*100:.2f}%)
                        - Histogram Similarity Score: {histogram_score:.4f} ({histogram_score*100:.2f}%)
                        - Overall Score: {overall_basic:.4f} ({overall_basic*100:.2f}%)

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
                        
                        # เรียก Gemini API
                        response = model.generate_content([prompt, image1, image2])
                        ai_analysis = response.text
                        
                        # แสดงผลการวิเคราะห์ AI
                        st.subheader("🤖 การวิเคราะห์โดย Gemini AI")
                        st.markdown(ai_analysis)
                        
                        # เก็บประวัติการวิเคราะห์
                        analysis_record = {
                            "timestamp": st.session_state.get("current_time", "Unknown"),
                            "basic_score": basic_score,
                            "color_score": color_score,
                            "histogram_score": histogram_score,
                            "overall_score": overall_basic,
                            "ai_analysis": ai_analysis
                        }
                        st.session_state.analysis_history.append(analysis_record)
                        
                except Exception as e:
                    error_message = f"Error processing request: {str(e)}"
                    st.error(f"❌ {error_message}")
                    
                    # Fallback response
                    st.subheader("📊 ผลการวิเคราะห์พื้นฐาน")
                    basic_score = calculate_basic_similarity(image1, image2)
                    color_score = calculate_color_similarity(image1, image2)
                    overall_score = (basic_score + color_score) / 2
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Basic Similarity", f"{basic_score:.3f}", f"{basic_score*100:.1f}%")
                    with col2:
                        st.metric("Color Similarity", f"{color_score:.3f}", f"{color_score*100:.1f}%")
                    with col3:
                        st.metric("Overall", f"{overall_score:.3f}", f"{overall_score*100:.1f}%")
                    
                    st.warning("⚠️ ไม่สามารถใช้ AI Analysis ได้ แต่ยังคงแสดงผลการวิเคราะห์พื้นฐาน")
            else:
                st.error("❌ Gemini model is not available")

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
        st.write("- **Histogram Similarity:** Color Distribution Correlation Analysis")
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

# ==============================
# Analysis History (Optional)
# ==============================
if st.session_state.analysis_history:
    with st.expander(f"📈 ประวัติการวิเคราะห์ ({len(st.session_state.analysis_history)} ครั้ง)"):
        for i, record in enumerate(reversed(st.session_state.analysis_history)):
            st.markdown(f"**การวิเคราะห์ครั้งที่ {len(st.session_state.analysis_history) - i}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Basic", f"{record['basic_score']:.3f}")
            with col2:
                st.metric("Color", f"{record['color_score']:.3f}")
            with col3:
                st.metric("Histogram", f"{record['histogram_score']:.3f}")
            with col4:
                st.metric("Overall", f"{record['overall_score']:.3f}")
            st.markdown("---")
