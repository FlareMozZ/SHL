import streamlit as st
import json
from model import qa

st.set_page_config(page_title="SHL Recommender", layout="wide")

def main():
    st.title("🎯 SHL Assessment Recommender")
    st.markdown("Enter job requirements to get relevant assessments")
    
    query = st.text_area("Job Description:", height=150)
    
    if st.button("Get Recommendations"):
        with st.spinner("Finding best matches..."):
            try:
                result = qa({"query": query})
                response = json.loads(result['result'])
                
                st.subheader(f"Top {len(response['recommendations'])} Assessments")
                
                for idx, rec in enumerate(response["recommendations"], 1):
                    with st.expander(f"{idx}. {rec['assessment_name']}"):
                        st.markdown(f"""
                        **🔗 URL**: [{rec['url']}]({rec['url']})  
                        **⏱ Duration**: {rec['duration']} minutes  
                        **🖥 Remote Testing**: {rec['remote_testing']}  
                        **📊 Test Type**: {rec['test_type']}  
                        **⭐ Relevance Score**: {rec['relevance_score']:.2f}/1.00  
                        **📝 Match Reason**: {rec['relevance_explanation']}
                        """)
                
                st.markdown("---")
                st.subheader("Source References")
                for doc in result['source_documents']:
                    st.markdown(f"- {doc.metadata['url']}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()