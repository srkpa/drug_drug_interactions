"""
side_effects.py
Predict DDI and their side effects

Handles the primary functions
"""


def hello(english=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    english : bool, Optional, default: True
        Set whether to say hello in french

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    if english:
        return "Hello world" 
    return "Bonjour le monde"


if __name__ == "__main__":
    print(hello())
