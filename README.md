
# Job Description:
1) we need qdrant vDB installed on our server: https://demo.treasurehunter.media/
we already have Python installed.


2) we need some api, where i can send a request:
Add this text string "fregf regregreg" with id=22322 to index/table of user #332
can be written on python, preferable to run with console, like /path/insert.py str="freg ergreggfregre" id=22322 user_id=332
answer on request: true(1) and false(0).


3) we need api to recive n count of relevant strings for specific user
like /path/get.py string="freg ergreggfregre" user_id=332 limit=5
answer json array:
id (the one we submitted with string) | string | score of proximity
id (the one we submitted with string) | string | score of proximity
id (the one we submitted with string) | string | score of proximity
id (the one we submitted with string) | string | score of proximity


We might have 1000 users, with different texts, hope it wont be a problem