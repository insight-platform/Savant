<dl>
    <dt>Versions</dt>
    {%- for version in versions %}
    <dd><a href="{{ get_url(version) }}">{{ version }}</a></dd>
    {%- endfor %}
</dl>
