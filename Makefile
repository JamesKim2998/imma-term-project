do_nothing:
	echo 'Do nothing'

convert_to_pdf:
	jupyter nbconvert --to html notebooks/Problem1.ipynb
	jupyter nbconvert --to html notebooks/Problem2.ipynb
	jupyter nbconvert --to html notebooks/Problem3.ipynb
	wkhtmltopdf notebooks/Problem1.html report/Problem1.pdf
	wkhtmltopdf notebooks/Problem2.html report/Problem2.pdf
	wkhtmltopdf notebooks/Problem3.html report/Problem3.pdf
	rm notebooks/Problem1.html
	rm notebooks/Problem2.html
	rm notebooks/Problem3.html

archive_report:
	zip -b report_group3.zip report
