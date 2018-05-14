#更改Jupyter notebook的工作空间(主目录)
<ol>
  <li>Jupyter的工作空间在其配置文件ipython_notebook_config.py中</li>
  <li>查找配置文件命令<br><code>jupyter notebook --generate-config</code></li>
  <li>更改<pre># The directory to use for notebooks.这决定了jupyter启动目录  
c.NotebookApp.notebook_dir = '/path/to/your/notebooks'</pre></li>
</ol>
