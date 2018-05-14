# 更改Jupyter notebook的工作空间(主目录)
<br><ol>
  <li>Jupyter的工作空间在其配置文件ipython_notebook_config.py中</li>
  <li>查找配置文件命令<br><code>jupyter notebook --generate-config</code></li>
  <li>更改<pre># The directory to use for notebooks.这决定了jupyter启动目录  
c.NotebookApp.notebook_dir = '/path/to/your/notebooks'</pre></li>
</ol>

# Juypter notebook命令  
<table>
  <tr><td align="center" width="500">%quickref</td><td align="center" width="500">显示 IPython 快速参考</td></tr>
  <tr><td align="center" width="500">%magic</td><td align="center" width="500">	显示所有魔术命令的详细文档</td></tr>
  <tr><td align="center" width="500">%debug</td><td align="center" width="500">从最新的异常跟踪的底部进入交互式调试器</td></tr>
  <tr><td align="center" width="500">%pdb</td><td align="center" width="500">在异常发生后自动进入调试器</td></tr>
  <tr><td align="center" width="500">%reset</td><td align="center" width="500">删除 interactive 命名空间中的全部变量</td></tr>
  <tr><td align="center" width="500">%run script.py</td><td align="center" width="500">执行 script.py</td></tr>
  <tr><td align="center" width="500">%prun statement</td><td align="center" width="500">通过 cProfile 执行对 statement 的逐行性能分析</td></tr>
  <tr><td align="center" width="500">%time statement</td><td align="center" width="500">测试 statement 的执行时间</td></tr>
  <tr><td align="center" width="500">%timeit statement</td><td align="center" width="500">多次测试 statement 的执行时间并计算平均值</td></tr>
  <tr><td align="center" width="500">%who、%who_ls、%whos</td><td align="center" width="500">显示 interactive 命名空间中定义的变量，信息级别/冗余度可变</td></tr>
  <tr><td align="center" width="500">%xdel variable</td><td align="center" width="500">删除 variable，并尝试清除其在 IPython 中的对象上的一切引用</td></tr>
  <tr><td align="center" width="500">!cmd</td><td align="center" width="500">在系统 shell 执行 cmd</td></tr>
  <tr><td align="center" width="500">output=!cmd args</td><td align="center" width="500">执行cmd 并赋值</td></tr>
  <tr><td align="center" width="500">%bookmark</td><td align="center" width="500">使用 IPython 的目录书签系统</td></tr>
  <tr><td align="center" width="500">%cd direcrory</td><td align="center" width="500">切换工作目录</td></tr>
  <tr><td align="center" width="500">%pwd</td><td align="center" width="500">返回当前工作目录（字符串形式）</td></tr>
  <tr><td align="center" width="500">%env</td><td align="center" width="500">返回当前系统变量（以字典形式）</td></tr>
</table>

# python安装第三方库  
  <code>pip install ..(例:pip install numpy)</code>
