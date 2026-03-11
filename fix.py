import pathlib
f = pathlib.Path('.github/workflows/tests.yml')
c = f.read_text(encoding='utf-8')
c = c.replace(
    '    - name: Upload coverage\n      uses: codecov/codecov-action@v3\n      with:\n        file: ./coverage.xml\n        fail_ci_if_error: false',
    '    - name: Upload coverage\n      uses: codecov/codecov-action@v3\n      with:\n        token: ${{ secrets.CODECOV_TOKEN }}\n        file: ./coverage.xml\n        fail_ci_if_error: false'
)
f.write_text(c, encoding='utf-8')
print("patched")
