import app as myapp
from flask import session

with myapp.app.test_client() as c:
    with c.session_transaction() as sess:
        sess['user_id'] = 1
        sess['role'] = 'teacher'
        sess['username'] = 'test_teacher'
    
    rv = c.get('/teacher/analytics')
    print("Status code:", rv.status_code)
    
    html = rv.data.decode('utf-8')
    # print the script block
    import re
    match = re.search(r'<script>(.*?)</script>', html, re.DOTALL)
    if match:
        print("Script block found!")
        print(match.group(1))
    else:
        print("No script block found in html. Len:", len(html))
        print("Tail of HTML:", html[-500:])
