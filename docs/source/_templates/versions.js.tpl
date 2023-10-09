$(function() {
    const latest = '{{ latest }}';

    // current docs version, eg. `0.2.5`
    const version = $('.version').text().trim();

    // add notice with link to the latest version
    if (version !== latest) {
        $('div.rst-content').prepend(
            '<label>' +
            '<input type="checkbox" class="alert-checkbox" autocomplete="off" />' +
            '<div class="alert admonition caution">' +
            '<i class="fa fa-close"></i>' +
            '<span>' +
            'This is the documentation for Savant version ' + version + '. ' +
            'The documentation for the latest stable version {{ latest }} ' +
            'can be found <a href="{{ pages_url }}/v{{ latest }}">here</a>.' +
	        '</span></div></label>'
	    );
    }

    // add versions to flyout menu
    $('.rst-other-versions').load('{{ pages_url }}/versions.html', function() {
        $('.rst-versions').show();
    });


});
