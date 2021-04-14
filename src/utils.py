def expect(condition: bool, exeption: BaseException):
    if not condition:
        raise exeption
