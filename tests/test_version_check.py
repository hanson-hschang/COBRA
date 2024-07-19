import re


def test_version():
    import cobra

    assert hasattr(cobra, "version")

    # Check if the version is a string
    assert isinstance(cobra.version, str)

    # Check if the version string is in the correct format
    assert re.match(r"\d+\.\d+\.\d+", cobra.version)
