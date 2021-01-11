# Mimetype

`mimetype` command is used to determine the mime type of a file.

Sometimes mimetype gets the wrong type. For example, both `.txt` and `.ply` are `text/plain`. To detemine mimetype by extension, there are a few methods:

### Method 1 `xdg-mime install`

Create an `xml` file containing the relevant information of the filetype, for example, in a file called `ply-mime.xml` type:
```
    <?xml version="1.0"?>  
<mime-info xmlns='http://www.freedesktop.org/standards/shared-mime-info'>  
    <mime-type type="application/extension-ply">  
        <comment>.ply file</comment>  
        <glob pattern="*.ply"/>  
    </mime-type>  
</mime-info>
```

Run `xdg-mime install ply-mime.xml`. Possibly need to restart.

### 2 `update-mime-database`

Create the `.xml` file as above and add it to `/usr/share/mime/application` and `/usr/share/mime/packages`. Run `sudo update-mime-database /usr/share/mime`
