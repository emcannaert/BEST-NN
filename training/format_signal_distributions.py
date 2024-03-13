import sys,os

	
def year_converter(year):
	year_dict = {"2015":"2016preAPV", "2016":"2016postAPV","2017":"2017","2018":"2018"}
	return year_dict[year]
if __name__=="__main__":
	years = ["2015","2016","2017","2018"]
	output_file = open("signal_distributions_latex_text.txt",'w')
	#samples = ["QCD","TTbarMC","STMC","SuuToChiChi"]
	plot_types = ["h_SJ_mass","h_disuperjet_mass"]
	plot_descriptions = [" superjet mass", "diSuperjet mass"]
	#regions = ["SR","CR","AT1b","AT0b"]
	region_descriptions = ["signal region", "control region", "1b anti-tag region", "0b anti-tag region"]
	#syst_names = []
	for year in years:
		for iii,plot_type in enumerate(plot_types):


		output_file.write("\begin{figure}[htbp]"+"\n")
		output_file.write(r"    \centering"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image "%() + r"}}"+"\n")
		output_file.write(r"\hfil"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image"%() + r"}}"+"\n")

		output_file.write(r"\medskip"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image"%() + r"}}"+"\n")
		output_file.write(r"\hfil"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image"%() + r"}}"+"\n")

		output_file.write(r"\medskip"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image"%() + r"}}"+"\n")
		output_file.write(r"\hfil"+"\n")
		output_file.write(r"\sidesubfloat[]{\includegraphics[width=0.4\textwidth]{" +  "example-image"%() +r"}}"+"\n")
		output_file.write(r"\caption{Main caption \dots}"+"\n")
		output_file.write(r"    \label{fig:myfigure}"+"\n")
		output_file.write(r"    \end{figure}"+"\n")







	output_file.close()
	print("Finished - output saved to event_weight_latex_text.txt")



#h_m_diSJ_ATShape_RatioPlot_<year>.png
#h_m_SJ1_ATShape_RatioPlot_<year>.png

			r"""
			output_file.write(r"\begin{figure" + "}\n")
			output_file.write(r"\subfloat[]{\includegraphics[width = 3in]" + "{plots/misc_plots/%s_SRShape_RatioPlot_%s.png"%(plot_type,year)+ r"}}"+"\n")
			output_file.write(r"\subfloat[]{\includegraphics[width = 3in]" + "{plots/misc_plots/%s_ATShape_combined_%s.png"%(plot_type,year)+ r"}}"+"\n")

			output_file.write(r"\caption{ " + "Combined MC backround SR/CR and AT1b/AT0b shapes as function of %s for %s. The SR/CR shape is shown in (a) and the AT1b/AT0b shape in (b). }"%(plot_descriptions[iii],year_converter(year) )+ "\n")
			output_file.write(r"\label{fig:" + "%s_shape_comparison_%s}\n"%(plot_type,year) ) 
			output_file.write(r"\end{figure}" + "\n")
			output_file.write("\n")
			output_file.write("\n")

			"""