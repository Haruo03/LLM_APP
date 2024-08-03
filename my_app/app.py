import streamlit as st
import hmac
import importlib
import sys
from pathlib import Path

# ディレクトリのsysパスを追加
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="Good App",
    page_icon=""
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # パスワードを保持しない
        else:
            st.session_state["password_correct"] = False

    # パスワードが検証された場合Trueを返す
    if st.session_state.get("password_correct", False):
        return True

    # パスワード入力
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Password incorrect")
    return False

if not check_password():
    st.stop()  # check_passwordがTrueでない場合は続行しない。

# ページ定義
pages = {
    "PDF": "contents.pdf_page",
    "Youtube": "contents.youtube_page",
    "Image": "contents.image_page",
    "Web site": "contents.website_page"
}

# サイドバー
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# 選択されたページを読み込み、表示
page = importlib.import_module(pages[selection])
page.main()