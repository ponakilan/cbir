import streamlit as st
from cbir import db_image_search
from PIL import Image


st.set_page_config(page_title="Image Search App", layout="wide", page_icon="🔍")


def main():
    st.title("Image Search Gallery")
    st.write("Upload a query image to find similar images.")

    with st.sidebar:
        st.header("Search Input")
        uploaded_file = st.file_uploader(
            "Upload an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Query Image")
            search_button = st.button(
                "Search", type="primary", use_container_width=True
            )
        else:
            search_button = False

    st.header("Search Results")

    if uploaded_file is not None and search_button:
        st.toast("Search initiated!")

        results = db_image_search(Image.open(uploaded_file).convert("RGB"))

        cols_per_row = 3

        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)

            for col, img_src in zip(cols, results[i : i + cols_per_row]):
                with col:

                    st.image(img_src, use_container_width=True)

    elif uploaded_file is None:
        st.info("Please upload an image in the sidebar to get started.")
    else:
        st.info("Click 'Search' in the sidebar to view results.")


if __name__ == "__main__":
    main()
