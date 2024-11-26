import streamlit as st
from main import process_news  # Import the `process_news` function from main.py

# Streamlit app interface
st.title("News AI App")
st.write("Analyze news articles and classify them as Pro or Con!")

# Input field for the query
query = st.text_input("Enter a topic to search for news:")

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
            # Pro Articles Section
            st.subheader(f"Pro Articles ({result_data['positive_count']})")
            for article in result_data["pro_articles"]:
                if article["url"]:  # Ensure only valid links are displayed
                    st.markdown(
                        f"""
                        **Source:** [{article['source']}]({article['url']})  
                        **Description:** {article['content']}  
                        """,
                        unsafe_allow_html=True,
                    )
                    st.divider()

            # Con Articles Section
            st.subheader(f"Con Articles ({result_data['negative_count']})")
            for article in result_data["con_articles"]:
                if article["url"]:  # Ensure only valid links are displayed
                    st.markdown(
                        f"""
                        **Source:** [{article['source']}]({article['url']})  
                        **Description:** {article['content']}  
                        """,
                        unsafe_allow_html=True,
                    )
                    st.divider()

            # Sentiment Analysis Stats
            st.subheader("Sentiment Analysis Stats")
            st.write(f"**Total Relevant Articles:** {result_data['total_count']}")
            st.write(f"**Positive Articles:** {result_data['positive_count']}")
            st.write(f"**Negative Articles:** {result_data['negative_count']}")