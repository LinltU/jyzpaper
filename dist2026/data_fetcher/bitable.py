import requests
import json

class Bitable:

    """
    A class to interact with Feishu Bitable API.

    This class provides methods to perform operations on Feishu Bitable, such as writing records to a table,
    searching for records, creating new tables, and sending messages. It requires setting up with a table ID,
    application ID, application secret, and wiki ID.

    Attributes:
        table_id (str): The ID of the Bitable table.
        app_id (str): The ID of the Feishu application.
        app_secret (str): The secret key of the Feishu application.
        wiki_id (str): The ID of the Feishu wiki space.
        app_token (str): The access token for the Feishu application.
        space_id (str): The ID of the Feishu space.
        tenant_access_token (str): The access token for the Feishu tenant.
    """

    def __init__(self, table_id, app_id, app_secret, wiki_id):
        """
        Initializes the Bitable class with the necessary credentials and IDs.

            Parameters:
            - table_id (str): The ID of the Bitable table to interact with.
            - app_id (str): The ID of the Feishu application.
            - app_secret (str): The secret key for the Feishu application.
            - wiki_id (str): The ID of the Feishu wiki space.
        """
        self.table_id = table_id
        self.app_id = app_id
        self.app_secret = app_secret
        self.app_token = None
        self.wiki_id = wiki_id
        self.tenant_access_token = None
        self.space_id = None
        self.get_app_token_and_space_id()

    def get_token(self):
        """
        Retrieves the tenant access token for the Feishu application.

        Returns:
        - str: The tenant access token.
        """
        url = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"
        payload = json.dumps({
            "app_id": self.app_id,
            "app_secret": self.app_secret
        })
        headers = {
            'Content-Type': 'application/json'
        }
        r = requests.request("POST", url, headers=headers, data=payload)
        json_data = json.loads(r.text)
        self.tenant_access_token = json_data['tenant_access_token']
        return self.tenant_access_token

    def write_in_bitable(self, index, cont):
        """
        Writes a record to the specified Bitable table.

        Parameters:
        - index (list): A list of field indices for the Bitable table.
        - cont (list): A list of field contents corresponding to the indices.

        Returns:
        - None: If the record is successfully written, otherwise prints an error message.
        """
        self.get_token()
        upload_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{str(self.app_token)}/tables/{str(self.table_id)}/records/"
        rec = self.process_post_data(index, cont)
        if rec is None:
            return
        rec = json.dumps(rec)
        # print(rec)
        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.post(upload_url, data=rec, headers=header)
        # print(r.text)

    def get_app_token_and_space_id(self):
        """
        Retrieves the application token and space ID using the wiki ID.

        Returns:
        - None: If the wiki ID is None, otherwise sets the app_token and space_id attributes.
        """
        if self.wiki_id is None:
            return

        self.get_token()
        # print(self.tenant_access_token)
        url = "https://open.feishu.cn/open-apis/wiki/v2/spaces/get_node?token="
        url = url + self.wiki_id
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        r = requests.request("GET", url, headers=header)
        json_data = json.loads(r.text)
        self.app_token = json_data['data']['node']['obj_token']
        self.space_id = json_data['data']['node']['space_id']

    @staticmethod
    def process_post_data(index, content):
        """
        Static method to process the post data into the format required by the Bitable API.

        Parameters:
        - index (list): A list of field indices for the Bitable table.
        - content (list): A list of field contents corresponding to the indices.

        Returns:
        - dict: A dictionary in the format required by the Bitable API for creating a record.
        - None: If the lengths of index and content do not match, prints an error message.
        """
        record = {"fields": {}}
        if len(index) == len(content):
            for i in range(len(index)):
                record["fields"][index[i]] = content[i]
            # print(record)
            return record
        else:
            print("The length of index should be the same as content.")
            return

    def add_to_wiki(self, space_id):
        """
        Adds the Bitable to the specified wiki space.

        This method moves the Bitable to the wiki space, making it part of the space's documentation.

        Parameters:
        - space_id (str): The ID of the wiki space to which the Bitable will be added.

        Returns:
        - None: The method does not return a value, but it performs the API request to add the Bitable to the wiki.
        """
        url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/{space_id}/nodes/move_docs_to_wiki"
        self.get_token()
        payload = json.dumps({
            "obj_type": "bitable",
            "obj_token": str(self.app_token)
        })
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.request("POST", url, headers=header, data=payload)
        # print(response.json())

    def duplicate_bitable(self, name):
        """
        Duplicates the Bitable with a new name.

        This method creates a copy of the Bitable without its content and returns a new Bitable instance
        with the details of the duplicated Bitable.

        Parameters:
        - name (str): The name for the new duplicated Bitable.

        Returns:
        - Bitable: A new Bitable instance representing the duplicated Bitable with the provided name.
        """
        # print("Duplicating...")
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.app_token}/copy"
        self.get_token()
        payload = json.dumps({
            "name": name,
            "without_content": True
        })

        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.request("POST", url, headers=header, data=payload)
        response = response.json()
        # print(response)
        # print("Duplicated.")
        new_app_token = response.get("data", {}).get("app", {}).get("app_token")
        new_url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{new_app_token}/tables"
        response = requests.request("GET", new_url, headers=header)
        response = response.json()
        new_table_id = response["data"]["items"][0]
        copy = Bitable(new_table_id, self.app_id, self.app_secret, None)
        copy.app_token = new_app_token
        copy.space_id = self.space_id
        # print(response)
        return copy

    def get_table_ids(self):
        """
         Retrieves the list of table IDs associated with the Bitable app.

         This method fetches the table IDs for the Bitable app identified by the app token.

         Returns:
         - list: A list of table IDs (strings) associated with the Bitable app.

         Raises:
         - Exception: If there is an error in fetching the table IDs, the exception will be propagated.
         """
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.app_token}/tables"
        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.request("GET", url, headers=header)
        response = response.json()
        # print(response)
        try:
            table_ids = response["data"]["items"]
            return table_ids
        except Exception as e:
            raise e

    def search_for_records(self, index, content):
        """
        Searches for records in the Bitable table that match the given criteria.

        This method performs a search on the Bitable table using the provided field indexes
        and corresponding content to find matching records.

        Parameters:
        - index (list of str): A list of field indexes to search against.
        - content (list of str): A list of content strings to match with the field indexes.

        Returns:
        - int: The number of records found that match the search criteria.

        Raises:
        - Exception: If the length of the index list does not match the length of the content list,
          an exception will be raised indicating that the input lists must be of the same length.
        """
        if len(index) == len(content):
            payload = {
                "filter": {
                    "conjunction": "and",
                    "conditions": []
                }
            }

            for i in range(len(index)):
                payload["filter"]["conditions"].append({
                            "field_name": index[i],
                            "operator": "is",
                            "value": [content[i]]
                        })
            self.get_token()
            header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
            payload = json.dumps(payload)
            url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records/search"
            response = requests.request("POST", url, headers=header, data=payload)
            response = response.json()
            count = response["data"]["total"]
            return count
        else:
            raise Exception("The length of index should be the same as content.")

    def create_new_table(self, name, index, data_types):
        """
        Creates a new Bitable table with the specified name and fields.

        Parameters:
        - name (str): The name of the new table.
        - index (list of str): A list of field names for the new table.
        - data_types (list of str): A list of data types corresponding to the field names.

        Returns:
        - Bitable: An instance of Bitable representing the newly created table.

        Raises:
        - Exception: If the data types provided are not valid or if the table creation fails.
        """
        data_map = {
            "文本": 1,
            "数字": 2,
            "单选": 3,
            "多选": 4,
            "日期": 5,
            "复选框": 7,
            "人员": 11,
            "电话号码": 13,
            "超链接": 15,
            "附件": 17,
            "单向关联": 18,
            "公式": 20,
            "双向关联": 21,
            "地理位置": 22,
            "群组": 23,
            "创建时间": 1001,
            "最后更新时间": 1002,
            "创建人": 1003,
            "修改人": 1004,
            "自动编号": 1005
        }

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.app_token}/tables?app_token="
        url = url + self.app_token

        payload = {
            "table": {
                "name": name,
                "fields": []
            }
        }

        for i in range(len(index)):
            if isinstance(data_types[i], str):
                data_types[i] = data_map[data_types[i]]
            if data_types[i] == 5:
                payload["table"]["fields"].append({
                    "field_name": index[i],
                    "type": 5,
                    "property": {
                        "date_formatter": "yyyy/MM/dd HH:mm"
                    }
                })
            else:
                payload["table"]["fields"].append({
                    "field_name": index[i],
                    "type": data_types[i]
                })

        # print(payload)
        payload = json.dumps(payload)

        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}

        response = requests.request("POST", url, headers=header, data=payload)
        response = response.json()
        new_table_id = response.get("data", {}).get("table_id")

        new_bitable = Bitable(new_table_id, self.app_id, self.app_secret, None)

        new_bitable.app_token = self.app_token
        new_bitable.space_id = self.space_id
        return new_bitable

    def batch_create(self, record_list):
        """
        Inserts a batch of records into the Bitable table.

        Parameters:
        - record_list (list of dict): A list of dictionaries where each dictionary represents
          a record with field names as keys and field values as values.

        Returns:
        - dict: A dictionary containing the result of the batch create operation.

        Raises:
        - Exception: If the number of records exceeds the limit or if the table is not found.
        """
        if len(record_list) >= 500:
            print("Too many records at a time!")

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records/batch_create"
        payload = json.dumps(record_list)
        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.post(url, data=payload, headers=header)
        response = response.json()
        # print(response)
        response_code = response.get("code")
        if response_code == 1254001:
            raise Exception("Table not found!")

    def send_message(self, message, open_id_list):
        """
        Sends a message to a single recipient or a list of recipients.

        Parameters:
        - message (str): The content of the message to be sent.
        - open_id_list (str or list of str): A single open ID or a list of open IDs of the recipients.

        Raises:
        - Exception: If the message sending fails or if the open_id_list parameter is of an incorrect type.
        """

        if isinstance(open_id_list, str):
            url = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id"
            payload = json.dumps({
                "receive_id": open_id_list,
                "msg_type": "text",
                "content": "{\"text\":\"" + message + "\"}"
            })
            self.get_token()
            header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
            response = requests.request("POST", url, headers=header, data=payload)
            response = response.json()
            response_code = response.get("code")
            if response_code != 0:
                raise Exception("Failed to convey message!")

        elif isinstance(open_id_list, list):
            url = "https://open.feishu.cn/open-apis/message/v4/batch_send/"
            payload = json.dumps({
                "open_ids": open_id_list,
                "msg_type": "text",
                "content": {
                    "text": message
                }
            })
            self.get_token()
            header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
            response = requests.request("POST", url, headers=header, data=payload)
            response = response.json()
            response_code = response.get("code")
            if response_code != 0:
                # print(response)
                raise Exception("Failed to convey message!")

        else:
            raise Exception("Wrong param: open_id_list. The type of this should be either list or str!")

    def get_all_in_space(self):
        """
        Retrieves all items in the specified space.

        Returns:
        - list: A list of items in the space.

        Raises:
        - Exception: If there is an error retrieving the items from the space.
        """
        url = f"https://open.feishu.cn/open-apis/wiki/v2/spaces/{self.space_id}/nodes"
        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.request("GET", url, headers=header)
        response = response.json()
        items = response.get("data", {}).get("items")
        return items

    def get_name_list(self, chat_id):
        """
        Retrieves all members in the specified chat group.

        Returns:
        - list: A list of members and their open ids in the space.

        Raises:
        - Exception: If there is an error retrieving the members from the group.
        """
        url = f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members"
        self.get_token()
        header = {"Content-Type": "application/json", "Authorization": "Bearer " + str(self.tenant_access_token)}
        response = requests.request("GET", url, headers=header)
        response = response.json()
        items = response.get("data", {}).get("items")
        return items

