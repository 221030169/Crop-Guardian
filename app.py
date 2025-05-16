import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Crop Disease Predictor", layout="centered")
# --- Background and Text Style ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #dfffd6;  /* Light green background */
    color: black;               /* Black text */
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);  /* Transparent Header */
}

[data-testid="stSidebar"] {
    background-color: #f0f9f0;  /* Slightly lighter green for Sidebar */
    color: black;
}

.stButton>button {
    background-color: #2e7d32 !important;  /* Dark green button */
    color: white !important;
}

.stDownloadButton>button {
    background-color: #2e7d32 !important;  /* Dark green button */
    color: white !important;
}

h1, h2, h3, h4, h5, h6 {
    color: black !important;  /* Force header text to black */
}

div[data-testid="stMarkdownContainer"] * {
    color: black !important;  /* Force Markdown text to black */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Title ---
st.markdown("<h1 style='color: black;'>🌾 Crop Disease Prediction App</h1>", unsafe_allow_html=True)

# --- Settings ---
IMG_SIZE = 224
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
WHEAT_CLASSES = ['Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust']

# --- Load Models ---
@st.cache_resource
def load_models():
    potato_model = tf.keras.models.load_model("potato_project/potato_disease_model.h5")
    wheat_model = tf.keras.models.load_model("wheat_project/wheat_disease_model.h5")
    return potato_model, wheat_model

potato_model, wheat_model = load_models()

# --- Tips Dictionary ---
def get_agri_tips(disease, state):
    tips_dict = {
    "Potato___Late_blight": {
        "Uttar Pradesh": """
        **English:**  
        - Use certified, disease-free seed tubers.  
        - Apply fungicides like Mancozeb or Metalaxyl regularly.  
        - Avoid overhead irrigation to prevent moisture on leaves.  
        - Ensure proper spacing to improve air circulation.  
        - Monitor weather forecasts and spray preventively during wet conditions.  

        **हिन्दी:**  
        - प्रमाणित, रोग मुक्त बीज कंदों का उपयोग करें।  
        - नियमित रूप से मैंकोज़ेब या मेटालैक्ज़िल जैसे कवकनाशी का छिड़काव करें।  
        - पत्तियों पर नमी रोकने के लिए ओवरहेड सिंचाई से बचें।  
        - वायु संचार सुधारने के लिए उचित दूरी बनाए रखें।  
        - नम मौसम के दौरान पूर्वानुमान की जाँच करें और रोकथाम के लिए छिड़काव करें।  
        """,
        "default": """
        **English:**  
        - Avoid overcrowding of plants to reduce disease spread.  
        - Ensure proper drainage to prevent water stagnation.  
        - Remove and destroy infected plants immediately.  

        **हिन्दी:**  
        - पौधों की अत्यधिक भीड़ से बचें ताकि रोग का प्रसार कम हो।  
        - जलभराव को रोकने के लिए उचित जल निकासी सुनिश्चित करें।  
        - संक्रमित पौधों को तुरंत हटा दें और नष्ट कर दें।  
        """
    },
    "Potato___Early_blight": {
        "Uttar Pradesh": """
        **English:**  
        - Practice crop rotation with non-host crops.  
        - Remove infected leaves promptly.  
        - Apply fungicides like Chlorothalonil or Copper-based sprays.  
        - Avoid wetting leaves during irrigation.  
        - Monitor for early signs and take preventive action.  

        **हिन्दी:**  
        - गैर-आतिथ्य फसलों के साथ फसल चक्रीकरण का अभ्यास करें।  
        - संक्रमित पत्तियों को तुरंत हटा दें।  
        - क्लोरोथैलोनिल या तांबा आधारित स्प्रे जैसे कवकनाशी का छिड़काव करें।  
        - सिंचाई के दौरान पत्तियों को गीला होने से बचाएं।  
        - शुरुआती संकेतों की निगरानी करें और रोकथाम के उपाय करें।  
        """,
        "default": """
        **English:**  
        - Ensure proper plant spacing to reduce leaf moisture.  
        - Apply fungicides during early growth stages.  
        - Remove fallen leaves and debris to prevent infection.  

        **हिन्दी:**  
        - पत्तियों की नमी को कम करने के लिए उचित दूरी पर पौधे लगाएं।  
        - शुरुआती विकास चरणों में कवकनाशी का छिड़काव करें।  
        - संक्रमण रोकने के लिए गिरे हुए पत्तों और मलबे को हटा दें।  
        """
    },
    "Potato___healthy": {
        "Uttar Pradesh": """
        **English:**  
        - Your plant is healthy!  
        - Continue regular monitoring and proper irrigation.  
        - Maintain soil fertility with balanced nutrients.  
        - Keep fields clean to avoid pest attraction.  

        **हिन्दी:**  
        - आपका पौधा स्वस्थ है!  
        - नियमित निगरानी और उचित सिंचाई जारी रखें।  
        - संतुलित पोषक तत्वों के साथ मिट्टी की उर्वरता बनाए रखें।  
        - कीटों को आकर्षित होने से रोकने के लिए खेतों को साफ रखें।  
        """,
        "default": """
        **English:**  
        - Ensure soil health and regular monitoring.  
        - Use organic fertilizers for better growth.  
        - Protect plants from pests and diseases.  

        **हिन्दी:**  
        - मिट्टी के स्वास्थ्य और नियमित निगरानी को सुनिश्चित करें।  
        - बेहतर वृद्धि के लिए जैविक उर्वरकों का उपयोग करें।  
        - कीटों और बीमारियों से पौधों की रक्षा करें।  
        """
    },
    "Wheat___Yellow_Rust": {
        "Rajasthan": """
        **English:**  
        - Apply fungicides such as Propiconazole or Tebuconazole during early stages.  
        - Remove volunteer wheat plants from the field.  
        - Avoid excess nitrogen fertilizer application.  
        - Rotate with non-cereal crops.  
        - Ensure proper irrigation without waterlogging.  

        **हिन्दी:**  
        - शुरुआती चरणों में प्रोपिकोनाज़ोल या टेबुकोनाज़ोल जैसे कवकनाशी का छिड़काव करें।  
        - खेत से अनावश्यक गेहूं के पौधों को हटा दें।  
        - अत्यधिक नाइट्रोजन उर्वरक का प्रयोग न करें।  
        - गैर-अनाज फसलों के साथ फसल चक्रीकरण का पालन करें।  
        - उचित सिंचाई सुनिश्चित करें, जलभराव से बचें।  
        """,
        "default": """
        **English:**  
        - Regularly monitor for yellow streaks on leaves.  
        - Apply fungicides preventively.  
        - Ensure good airflow between plants.  

        **हिन्दी:**  
        - पत्तियों पर पीली धारियों के लिए नियमित निगरानी करें।  
        - रोकथाम के लिए कवकनाशी का छिड़काव करें।  
        - पौधों के बीच अच्छे वायु प्रवाह को सुनिश्चित करें।  
        """
    },
    "Wheat___Brown_Rust": {
        "Rajasthan": """
        **English:**  
        - Apply appropriate fungicides like Mancozeb or Propiconazole.  
        - Remove infected leaves and weeds.  
        - Maintain optimal plant spacing.  
        - Irrigate only during non-susceptible growth stages.  

        **हिन्दी:**  
        - मैंकोज़ेब या प्रोपिकोनाज़ोल जैसे उचित कवकनाशी का छिड़काव करें।  
        - संक्रमित पत्तियों और खरपतवारों को हटा दें।  
        - पौधों के बीच उचित दूरी बनाए रखें।  
        - संवेदनशील विकास चरणों के दौरान सिंचाई से बचें।  
        """,
        "default": """
        **English:**  
        - Prune infected parts and ensure field hygiene.  
        - Apply copper-based sprays.  
        - Rotate crops to break disease cycle.  

        **हिन्दी:**  
        - संक्रमित हिस्सों को काटें और खेत की स्वच्छता बनाए रखें।  
        - तांबा आधारित स्प्रे का छिड़काव करें।  
        - बीमारी के चक्र को तोड़ने के लिए फसलों का चक्रीकरण करें।  
        """
    },
    "Wheat___Healthy": {
        "Rajasthan": """
        **English:**  
        - Your wheat crop is healthy!  
        - Regular monitoring and optimal irrigation are advised.  
        - Use organic mulching to maintain moisture.  

        **हिन्दी:**  
        - आपकी गेहूं की फसल स्वस्थ है!  
        - नियमित निगरानी और उचित सिंचाई करें।  
        - नमी बनाए रखने के लिए जैविक मल्चिंग का उपयोग करें।  
        """
    }
}

    return tips_dict.get(disease, {}).get(state, tips_dict.get(disease, {}).get("default", "No tips available."))

# --- State Selection ---
selected_state = st.selectbox("Select your State:", ["Uttar Pradesh", "Rajasthan"])

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    if selected_state == "Uttar Pradesh":
        model = potato_model
        classes = POTATO_CLASSES
    else:
        model = wheat_model
        classes = WHEAT_CLASSES

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    predicted_class = classes[pred_index]
    confidence = np.max(predictions) * 100

    st.success(f"🧠 **Predicted Disease**: {predicted_class}")
    st.info(f"📊 **Confidence**: {confidence:.2f}%")

    # --- Show Tips ---
    if st.button("💡 Get AI Tips"):
        with st.spinner("Fetching expert tips..."):
            tips = get_agri_tips(predicted_class, selected_state)

            # Display suggestions with a lighter background and proper formatting
            st.markdown(
                f"""
                <div style='color:#000; backstreaststreamlitground-color:#e6ffe6; padding:15px; border-radius:8px;'>
                    {tips}
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- Download Report ---
    if st.button("📄 Download Report"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
📋 Disease Prediction Report

🕒 Date: {timestamp}
🌍 State: {selected_state}
🧠 Predicted Disease: {predicted_class}
📊 Confidence: {confidence:.2f}%
"""
        st.download_button("Download Report", report, file_name="prediction_report.txt")
