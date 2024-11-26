import streamlit as st
from main import process_news  # Import the updated `process_news` function from main.py
import matplotlib.pyplot as plt

# Configure the Streamlit app
st.set_page_config(page_title="News AI App", page_icon="📰", layout="wide")

# App title and description
st.title("📰 News AI App")
st.write("Analyze news articles, classify them as Pro or Con, and view overall summaries!")

# Input field for the query
query = st.text_input("🔍 Enter a topic to search for news:")

# Search button
if st.button("Search"):
    if query:
        with st.spinner(f"Fetching and analyzing news for '{query}'..."):
            # Process the query using the `process_news` function
            result_data = process_news(query)

        # Display results
        if "error" in result_data:
            st.error(f"Error: {result_data['error']}")
        else:
            # Layout: Two columns for Positive and Negative Summaries
            st.subheader("Overall Summaries")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Positive Articles Summary")
                st.write(result_data["positive_summary"])

            with col2:
                st.markdown("### Negative Articles Summary")
                st.write(result_data["negative_summary"])

            # Divider
            st.divider()

            # Pro Articles Section
            st.subheader(f"🌟 Pro Articles ({result_data['positive_count']})")
            for article in result_data["pro_articles"]:
                if article["url"]:  # Ensure only valid links are displayed
                    with st.container():
                        st.markdown(
                            f"""
                            **Source:** [{article['source']}]({article['url']})  
                            **Description:** {article['content']}  
                            """,
                            unsafe_allow_html=True,
                        )
                        if "keywords" in article:
                            st.markdown(f"**Keywords:** {', '.join(article['keywords'])}")
                        st.divider()

            # Con Articles Section
            st.subheader(f"⚡ Con Articles ({result_data['negative_count']})")
            for article in result_data["con_articles"]:
                if article["url"]:  # Ensure only valid links are displayed
                    with st.container():
                        st.markdown(
                            f"""
                            **Source:** [{article['source']}]({article['url']})  
                            **Description:** {article['content']}  
                            """,
                            unsafe_allow_html=True,
                        )
                        if "keywords" in article:
                            st.markdown(f"**Keywords:** {', '.join(article['keywords'])}")
                        st.divider()

            # Sentiment Analysis Stats with Visualization
            positive_count = result_data["positive_count"]
            negative_count = result_data["negative_count"]

            labels = ["Positive Articles", "Negative Articles"]
            sizes = [positive_count, negative_count]
            colors = ["#4CAF50", "#FF5252"]

            st.subheader("📊 Sentiment Analysis Stats")
            
            # Create a pie chart to visualize sentiment distribution
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
            
            # Display the pie chart in Streamlit
            st.pyplot(fig)

            # Display sentiment stats as metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Positive Articles", value=positive_count)
            with col2:
                st.metric(label="Negative Articles", value=negative_count)

            total_articles = result_data["total_count"]
            st.write(f"**Total Relevant Articles Analyzed:** {total_articles}")