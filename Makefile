do_nothing:
	echo 'Do nothing'

convert_to_pdf:
	jupyter nbconvert --to html notebooks/Problem1.ipynb
	jupyter nbconvert --to html notebooks/Problem2.ipynb
	jupyter nbconvert --to html notebooks/Problem3.ipynb
	wkhtmltopdf notebooks/Problem1.html Problem1.pdf
	wkhtmltopdf notebooks/Problem2.html Problem2.pdf
	wkhtmltopdf notebooks/Problem3.html Problem3.pdf
