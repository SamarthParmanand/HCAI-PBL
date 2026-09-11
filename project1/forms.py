from django import forms


class CSVUploadForm(forms.Form):
    """The upload control.

    The Bootstrap class is set here rather than in the template: the widget is
    part of the form definition, and the template only renders what it is given.
    """

    file = forms.FileField(
        label="Select a CSV file",
        widget=forms.ClearableFileInput(attrs={
            "class": "form-control",
            "accept": ".csv,text/csv",
        }),
    )
