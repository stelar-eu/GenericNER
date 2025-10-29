import streamlit as st

# Set page config
st.set_page_config(
    page_title="STELAR NER-EL System",
    page_icon="📊",
    layout="wide"
)

# Create navigation - dropdown only, no tabs
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Select a page from the dropdown below:")

page = st.sidebar.selectbox(
    "Choose a page:", 
    ["Dashboard", "Evaluation"],
    help="Use this dropdown to navigate between different pages of the application"
)

st.sidebar.markdown("---")

# Import and run the selected page
if page == "Dashboard":
    import pages.dashboard as dashboard
    dashboard.show()
elif page == "Evaluation":
    import pages.evaluation as evaluation
    evaluation.show()
