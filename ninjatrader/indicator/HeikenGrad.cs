#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using SharpDX;
#endregion

//This namespace holds Indicators in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Indicators
{
	public class HeikenGrad : Indicator
	{
        private CustomHeikenAshi Heiken;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description							= @"Calculates a gradient based on HeikenAshi bars.";
				Name								= "HeikenGrad";
				Calculate							= Calculate.OnEachTick;
				DrawOnPricePanel					= false;
				DrawVerticalGridLines				= true;
				IsSuspendedWhileInactive			= false;
				DrawHorizontalGridLines 			= false;
				BarsRequiredToPlot					= 1;
				
				Period = 2;
				
				AddPlot(Brushes.RoyalBlue, "Gradient");
				AddPlot(Brushes.Yellow, "Average Gradient");
			}
			else if (State == State.DataLoaded)
			{
				Heiken = CustomHeikenAshi();
				
				Draw.HorizontalLine(this, "Zero", Threshold, Brushes.WhiteSmoke, false);
				Draw.HorizontalLine(this, "Upper Threshold", Threshold, Brushes.DarkCyan, false);
				Draw.HorizontalLine(this, "Lower Threshold", 0-Threshold, Brushes.DarkCyan, false);
			}
		}

		protected override void OnBarUpdate()
		{
			double grad = Gradient();
			
			Grad[0] = grad;
			AvgGrad[0] = grad / Period;
		}
		
		private double Gradient()
		{
			double gradient = 0;
			if (Period >= CurrentBar + 1)
				return gradient;
			
			foreach (int i in Enumerable.Range(0, Period-1))
			{
				double hClose = (Heiken.HAClose[i]+ Heiken.HAOpen[i] + Heiken.HAHigh[i]+ Heiken.HALow[i]) / 4;
				double hOpen = (Heiken.HAClose[i+1]+ Heiken.HAOpen[i+1]) / 2;
				
				gradient += hClose - hOpen;
			}
			
			return gradient;
		}

		#region Properties
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Period", GroupName = "Parameters", Order = 0)]
		public int Period
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 1)]
		public double Threshold
		{ get; set; }
		
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> Grad
		{
			get { return Values[0]; }
		}
		
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> AvgGrad
		{
			get { return Values[1]; }
		}
		#endregion
	
		}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private HeikenGrad[] cacheHeikenGrad;
		public HeikenGrad HeikenGrad(int period, double threshold)
		{
			return HeikenGrad(Input, period, threshold);
		}

		public HeikenGrad HeikenGrad(ISeries<double> input, int period, double threshold)
		{
			if (cacheHeikenGrad != null)
				for (int idx = 0; idx < cacheHeikenGrad.Length; idx++)
					if (cacheHeikenGrad[idx] != null && cacheHeikenGrad[idx].Period == period && cacheHeikenGrad[idx].Threshold == threshold && cacheHeikenGrad[idx].EqualsInput(input))
						return cacheHeikenGrad[idx];
			return CacheIndicator<HeikenGrad>(new HeikenGrad(){ Period = period, Threshold = threshold }, input, ref cacheHeikenGrad);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.HeikenGrad HeikenGrad(int period, double threshold)
		{
			return indicator.HeikenGrad(Input, period, threshold);
		}

		public Indicators.HeikenGrad HeikenGrad(ISeries<double> input , int period, double threshold)
		{
			return indicator.HeikenGrad(input, period, threshold);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.HeikenGrad HeikenGrad(int period, double threshold)
		{
			return indicator.HeikenGrad(Input, period, threshold);
		}

		public Indicators.HeikenGrad HeikenGrad(ISeries<double> input , int period, double threshold)
		{
			return indicator.HeikenGrad(input, period, threshold);
		}
	}
}

#endregion
