
class ResponsesMessage:
    def input_text_message(self, text):
        return {
            "type": "input_text",
            "text": text
        }
    def file_message(self, file_data, filename):
        return {
            "type": "input_file",
            "file_data": file_data,
            "filename": filename
        }
    def input_file_id(self, file_id):
        return {
            "type": "input_file",
            "file_id": file_id
        }
    def output_text_message(self, text):
        return {
            "type": "output_text",
            "text": text
        }