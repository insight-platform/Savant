$(function() {
    const latest = '{{ latest }}';
    console.log(latest);

    // current docs version, eg. `0.2.5`
    const version = $('.version').text().trim();
    console.log(version);

    // add versions to flyout menu
    $('.rst-other-versions').load('{{ pages_url }}/versions.html', function() {
        $('.rst-versions').show();
    });


});
