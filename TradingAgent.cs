
#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using System.Net.Http;
using Newtonsoft.Json;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
#endregion


namespace NinjaTrader.NinjaScript.Strategies
{
	public class TradingAgent : Strategy
	{	
		private DateTime firstDateTime;
		private DateTime lastDateTime;
		private bool isExecuted;
		private bool isLastSequence;
		private int numBar;
		private int sequenceLength = 10;

		public class RequestData
        {
            public List<double> progressList = new List<double>();
            public List<double> openList = new List<double>();
            public List<double> highList = new List<double>();
            public List<double> lowList = new List<double>();
            public List<double> closeList = new List<double>();
            public List<double> volumeList = new List<double>();
            public int contracts;
            public double entryPrice;
            public double balance;
		}

		private RequestData data = new RequestData();

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description									= @"Enter the description for your new custom Strategy here.";
				Name										= "TradingAgent";
				Calculate									= Calculate.OnBarClose;
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 0;
				IsInstantiatedOnEachOptimizationIteration	= true;
			}
		}
		protected override async void OnBarUpdate()
		{
		    if (State == State.Historical)
                return;

			numBar = CurrentBar;
			if(!isExecuted)
			{
				firstDateTime = Time[0];
				lastDateTime = Time[0].Date.AddHours(18);
				isExecuted = true;
			}
			
			CalcProgress();
			
			if((numBar % sequenceLength == 0 && numBar != 0) || isLastSequence)
			{
				CollectData();
				SendHttpRequest(Newtonsoft.Json.JsonConvert.SerializeObject(data));
				ClearData();
			}
		}
		
		private void CollectData()
		{			
			for (int i = 0; i < sequenceLength; i++)
			{
				data.openList.Add(Open[i]);
				data.highList.Add(High[i]);
				data.lowList.Add(Low[i]);
				data.closeList.Add(Close[i]);
				data.volumeList.Add(Volume[i]);
			}
			data.contracts = Position.Quantity;
			data.entryPrice = Position.AveragePrice;
			data.balance = GetAccountBalance();
		}
		
		private void ClearData()
		{
			data.progressList.Clear();
			data.openList.Clear();
			data.highList.Clear();
			data.lowList.Clear();
			data.closeList.Clear();
			data.volumeList.Clear();
		}
		
		private double GetAccountBalance()
		{
			Account a = Account.All.First(t => t.Name == "Sim101");
			double value = a.Get(AccountItem.CashValue, Currency.UsDollar);
			return value;
		}
		
		//Calculate the current progress
		private void CalcProgress()
		{
			double firstHours = (firstDateTime - firstDateTime.Date).TotalHours;
			double lastHours = (lastDateTime - lastDateTime.Date).TotalHours;
			double hourDiff = lastHours - firstHours;
			data.progressList.Add(numBar / (hourDiff * 60));
			
			if(data.progressList.Last() == 1)
			{
				ResetParameters();
			}
		}
		
		//Resets Parameters because new Trading Day begins
		private void ResetParameters()
		{
			numBar = 0;
			isExecuted = false;
			isLastSequence = true;
		}
		
		private async void SendHttpRequest(string json)
        {
            using (var httpClient = new HttpClient())
            {
                try
                {
                    var url = "http://localhost:8000/predict";
                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(url, content);

                    if (response.IsSuccessStatusCode)
                    {
                        var responseContent = await response.Content.ReadAsStringAsync();
                        Print("POST request sent successfully. Response: " + responseContent);
						
		                dynamic parsedResponse = Newtonsoft.Json.JsonConvert.DeserializeObject(responseContent);
		                string action = parsedResponse.action;
		                int amount = parsedResponse.amount;
						
		                switch (action)
		                {
		                    case "LONG":
								ExitPosition();
		                        EnterLong(amount);
		                        Print("Entered a long position with amount: " + amount);
		                        break;
		                    case "SHORT":
								ExitPosition();
		                        EnterShort(amount);
		                        Print("Entered a short position with amount: " + amount);
		                        break;
		                    case "STAY":
		                        Print("No action taken. Stay in current position.");
		                        break;
		                    case "EXIT":
		                        ExitPosition();
		                        Print("Exited all positions.");
		                        break;
		                    default:
		                        Print("Invalid action received from the response.");
		                        break;
		                }
                    }
                    else
                    {
                        Print("POST request failed. Response status code: " + response.StatusCode);
                    }
                }
                catch (Exception ex)
                {
                    Print("Exception occurred: " + ex.Message);
                }
            }
        }
		
		private void ExitPosition()
		{
			ExitLong();
			ExitShort();	
		}
		
		#region Properties



		#endregion
	}
}