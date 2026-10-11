"""Shared UI helper functions for Streamlit application.

Provides layout helpers, base64 animated GIF renderers, and custom CSS injectors.
"""

import base64
import pathlib
from typing import Any, Union

import streamlit as st


def stretch(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke Streamlit element layout with stretch width falling back gracefully.

    Args:
        fn: Streamlit display callable (e.g. st.pyplot, st.dataframe).
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Result of the Streamlit element call.
    """
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, ValueError, st.errors.StreamlitAPIException):
        return fn(*args, use_container_width=True, **kwargs)


def show_gif(gif_bytes: bytes, max_width: int = 900) -> None:
    """Render in-memory animated GIF using an HTML base64 data URI.

    Args:
        gif_bytes: Raw bytes of the GIF image.
        max_width: Maximum display width in CSS pixels.
    """
    b64 = base64.b64encode(gif_bytes).decode("ascii")
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" style="width: 100%; max-width: {max_width}px; border-radius: 6px;">',
        unsafe_allow_html=True,
    )


def inject_css(path: Union[str, pathlib.Path] = "assets/style.css") -> None:
    """Read stylesheet if it exists and is not empty and inject into Streamlit.

    Args:
        path: Filepath to CSS stylesheet.
    """
    css_file = pathlib.Path(path)
    if css_file.exists():
        css_text = css_file.read_text(encoding="utf-8").strip()
        if css_text:
            st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
