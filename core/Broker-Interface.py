import fxcmpy as forex_broker
import json

API_TOKEN = json.load(open(r"D:\Data\markets\settings\settings.json"))["FXCM"]["REST API Token"]


def main():
	client = forex_broker.fxcmpy(config_file=r"D:\Data\markets\settings\fxcm.cfg")
	df = client.get_accounts_summary()
	print(df["balance"])


if __name__ == '__main__':
	main()
