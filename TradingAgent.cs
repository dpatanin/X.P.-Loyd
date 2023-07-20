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
	// TODO: Placeholder for data class
	public class MockData
    {
        public List<double> progress { get; set; }
        public List<double> open { get; set; }
        public List<double> high { get; set; }
        public List<double> low { get; set; }
        public List<double> close { get; set; }
        public List<double> volume { get; set; }
        public int contracts { get; set; }
        public int entryPrice { get; set; }
        public int balance { get; set; }
    }
	
	public class TradingAgent : Strategy
	{
		// TODO: Placeholder to test sending requests 
		private MockData data = new MockData()
		{
			progress = new List<double> {0.111,0.112,0.113,0.114,0.115,0.116,0.117,0.118,0.119,0.2},
			open = new List<double> {0,25,0,12.5,-100,-25,-150,0,250,200},
			high = new List<double> {12.5,50,25,12.5,-75,0,-50,12.5,275,250},
			low = new List<double> {-25,0,-50,0,-200,-50,-200,0,200,100},
			close = new List<double> {25,0,12.5,-4100,-25,-50,0,250,200,500},
			volume = new List<double> {788,122,850,657,234,888,1453,456,654,453},
			contracts = 5,
			entryPrice = 3500,
			balance = 10000
		};

		private static readonly HttpClient client = new HttpClient();

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"Description";
				Name = "TradingAgent";
				Calculate = Calculate.OnBarClose;
				EntriesPerDirection = 1;
				EntryHandling = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds = 30;
				IsFillLimitOnTouch = false;
				MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution = OrderFillResolution.Standard;
				Slippage = 0;
				StartBehavior = StartBehavior.WaitUntilFlat;
				TimeInForce = TimeInForce.Gtc;
				TraceOrders = false;
				RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling = StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade = 10;

				IsInstantiatedOnEachOptimizationIteration = true;
			}
		}
		
		protected override void OnBarUpdate()
		// TODO: Placeholder for when to send requests 
		{
			if (State == State.Historical)
				return;
			
			if (IsFirstTickOfBar)
			{
				SendHttpRequest(Newtonsoft.Json.JsonConvert.SerializeObject(data));
			}
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
		                        EnterLong(amount);
		                        Print("Entered a long position with amount: " + amount);
		                        break;
		                    case "SHORT":
		                        EnterShort(amount);
		                        Print("Entered a short position with amount: " + amount);
		                        break;
		                    case "STAY":
		                        Print("No action taken. Stay in current position.");
		                        break;
		                    case "EXIT":
		                        ExitLong();
								ExitShort();
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
		
	}
}