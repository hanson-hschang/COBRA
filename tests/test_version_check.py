import re


def test_version():
    import cobr2

    assert hasattr(cobr2, "version")

    # Check if the version is a string
    assert isinstance(cobr2.version, str)

    # Check if the version string is in the correct format
    assert re.match(r"\d+\.\d+\.\d+", cobr2.version)
