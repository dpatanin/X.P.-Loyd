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
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion
namespace NinjaTrader.NinjaScript.Strategies
{
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
		private SMA smaFast;
		private SMA smaSlow;
		
		private Timer timer; // Timer object to trigger the request
        private bool isFirstMinute = true; // Flag to send the request on the first minute
		
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
			else if (State == State.DataLoaded)
			{
				smaFast = SMA(10);
				smaSlow = SMA(25);

				smaFast.Plots[0].Brush = Brushes.Azure;
				smaSlow.Plots[0].Brush = Brushes.Crimson;

				AddChartIndicator(smaFast);
				AddChartIndicator(smaSlow);
				
				InitializeTimer();
			}
		}
		
		private void InitializeTimer()
        {
            // Calculate the time until the next minute
            DateTime now = Time[0];
            DateTime nextMinute = new DateTime(now.Year, now.Month, now.Day, now.Hour, now.Minute + 1, 0);
            TimeSpan timeUntilNextMinute = nextMinute - now;

            // Set up the timer to trigger the request
            timer = new Timer(OnTimerCallback, null, timeUntilNextMinute, TimeSpan.FromMinutes(1));
        }
		
		protected override void OnBarUpdate()
		{
			if (CurrentBar == BarsRequiredToTrade)
				return;

			if (CrossAbove(smaFast, smaSlow, 1))
				EnterLong();
			else if (CrossBelow(smaFast, smaSlow, 1))
				EnterShort();
		}
		
		private void OnTimerCallback(object state)
        {
            DateTime now = Time[0];

            // Check if it's the first minute or a subsequent minute
            if (isFirstMinute || now.Second == 0)
            {
                // Reset the flag after the first minute
                isFirstMinute = false;

                // Send the HTTP request
                SendHttpRequest(Newtonsoft.Json.JsonConvert.SerializeObject(data));
            }
        }

		private async void SendHttpRequest(string json)
        {
            using (var httpClient = new HttpClient())
            {
                try
                {
                    // Define the URL to send the request to
                    var url = "http://localhost:8000/predict";

                    // Create the HTTP content with the JSON body
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    // Send the POST request
                    var response = await httpClient.PostAsync(url, content);

                    // Check if the request was successful
                    if (response.IsSuccessStatusCode)
                    {
                        // Get the response content as a string
                        var responseContent = await response.Content.ReadAsStringAsync();

                        // Print the response content
                        Print("POST request sent successfully. Response: " + responseContent);
                    }
                    else
                    {
                        // Request failed
                        Print("POST request failed. Response status code: " + response.StatusCode);
                    }
                }
                catch (Exception ex)
                {
                    // Handle any exceptions
                    Print("Exception occurred: " + ex.Message);
                }
            }
        }
	}
}