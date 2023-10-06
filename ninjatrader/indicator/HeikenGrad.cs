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
		private EMA EHClose;
		private EMA EHOpen;
		
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
				Smooth = 1;
				
				AddPlot(Brushes.RoyalBlue, "Gradient");
				AddPlot(Brushes.Yellow, "Average Gradient");
			}
			else if (State == State.DataLoaded)
			{
				Heiken = CustomHeikenAshi();
				EHClose = EMA(Heiken.HAClose, Smooth);
				EHOpen = EMA(Heiken.HAOpen, Smooth);
				
				Draw.HorizontalLine(this, "Zero", 0, Brushes.WhiteSmoke, false);
				Draw.HorizontalLine(this, "Upper Threshold", Threshold, Brushes.DarkCyan, false);
				Draw.HorizontalLine(this, "Lower Threshold", 0-Threshold, Brushes.DarkCyan, false);
			}
		}

		protected override void OnBarUpdate()
		{
			Default[0] = Gradient(1);
			Avg[0] = Gradient(Period) / Period;
		}
		
		private double Gradient(int bars)
		{
			double gradient = 0;
			if (bars >= CurrentBar)
				return gradient;
			
			foreach (int i in Enumerable.Range(0, bars))
				gradient += EHClose[i] - EHOpen[i];
			
			return gradient;
		}

		#region Properties
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Period", GroupName = "Parameters", Order = 0)]
		public int Period
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Smooth", GroupName = "Parameters", Order = 1)]
		public int Smooth
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 2)]
		public double Threshold
		{ get; set; }
		
		[Browsable(false)]
		[XmlIgnore()]
		public Series<double> Default
		{
			get { return Values[0]; }
		}
		
		[Browsable(false)]
		[XmlIgnore]
		public Series<double> Avg
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
		public HeikenGrad HeikenGrad(int period, int smooth, double threshold)
		{
			return HeikenGrad(Input, period, smooth, threshold);
		}

		public HeikenGrad HeikenGrad(ISeries<double> input, int period, int smooth, double threshold)
		{
			if (cacheHeikenGrad != null)
				for (int idx = 0; idx < cacheHeikenGrad.Length; idx++)
					if (cacheHeikenGrad[idx] != null && cacheHeikenGrad[idx].Period == period && cacheHeikenGrad[idx].Smooth == smooth && cacheHeikenGrad[idx].Threshold == threshold && cacheHeikenGrad[idx].EqualsInput(input))
						return cacheHeikenGrad[idx];
			return CacheIndicator<HeikenGrad>(new HeikenGrad(){ Period = period, Smooth = smooth, Threshold = threshold }, input, ref cacheHeikenGrad);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.HeikenGrad HeikenGrad(int period, int smooth, double threshold)
		{
			return indicator.HeikenGrad(Input, period, smooth, threshold);
		}

		public Indicators.HeikenGrad HeikenGrad(ISeries<double> input , int period, int smooth, double threshold)
		{
			return indicator.HeikenGrad(input, period, smooth, threshold);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.HeikenGrad HeikenGrad(int period, int smooth, double threshold)
		{
			return indicator.HeikenGrad(Input, period, smooth, threshold);
		}

		public Indicators.HeikenGrad HeikenGrad(ISeries<double> input , int period, int smooth, double threshold)
		{
			return indicator.HeikenGrad(input, period, smooth, threshold);
		}
	}
}

#endregion
