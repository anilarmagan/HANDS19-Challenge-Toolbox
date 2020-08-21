# modifier: Anil Armagan
# contact: a.armagan@imperial.ac.uk
# desc: this file implements a vertex sharer and a ortho renderer class for MANO

"""
Functions for rendering a single MANO model to image
"""
import moderngl
import numpy as np

vertex_shader = '''
    #version 330 core
    in vec3 in_vert;
    uniform mat4 mvp;

    void main() {
        gl_Position =  vec4(in_vert, 1.0);
    }
'''

class HandRenderer:
    def __init__(self, faces):
        """
        Class for rendering a hand from parameters
        """
        self.num_vertices = 778
        self.znear = 0.01
        self.zfar = 0.3 # 300mm bbox size

        # opengl settings
        try:
            self.ctx = moderngl.create_standalone_context()
        except moderngl.Error as err:
            print(err.function, err.filename, err.line)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader
        )

        # # initialize buffers, HandRenderer object should be initialized
        self.vboPos = self.ctx.buffer(reserve=self.num_vertices * 3 * 4, dynamic=True)
        self.ibo = self.ctx.buffer(faces.astype('i4').tobytes())

        vao_content = [
            # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
            (self.vboPos, '3f', 'in_vert')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)


    def init_buffers(self, image_w, image_h):

        self.width, self.height = image_w, image_h
        self.rbo_color = self.ctx.renderbuffer((image_w, image_h))
        self.rbo_depth = self.ctx.depth_texture((image_w, image_h), alignment=1)
        self.fbo = self.ctx.framebuffer(color_attachments=self.rbo_color, depth_attachment=self.rbo_depth)

    def render_mano(self, vertices):
        """
        Render Mano vertices on an image
        :param vertices: vertices of model render
        """

        self.vboPos.write(vertices.astype('f4').tobytes())

        # Rendering
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.vao.render()

        # z buffer read
        depth = np.frombuffer(self.rbo_depth.read(alignment=1), dtype=np.dtype('f4')).reshape((self.height, self.width)) * self.zfar
        return depth * 1000. # mm

    def __del__(self):
        self.prog.release()
        self.vboPos.release()
        self.ibo.release()
        self.vao.release()
        self.fbo.release()
